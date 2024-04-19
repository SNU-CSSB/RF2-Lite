import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from data_loader import MSAFeaturize, MSABlockDeletion
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, get_init_xyz
from trFold import TRFold
NBIN = [37, 37, 37, 19]

MODEL_PARAM ={
        "n_extra_block": 4,
        "n_main_block": 8,
        "n_ref_block": 4,
        "d_msa"           : 256 ,
        "d_pair"          : 128,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 4,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 64,
        "p_drop"       : 0.0,
        }

SE3_param = {
        "num_layers"    : 3,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 32,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 32,
        "div": 4,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param'] = SE3_param

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument("-list", required=True, help="The file containing a list of input MSA, lengths, and output prefix e.g. P77499_P77689.fas 248 423 P77499_P77689") 
    parser.add_argument("-p", required=True, help="The CUDA selection") 
    parser.add_argument('-maxlat', type=int, default=128, help="Maximum number of latent sequences [128]")
    parser.add_argument('-maxseq', type=int, default=1024, help="Maximum number of extra sequences [1024]")
    parser.add_argument('-maxcycle', type=int,  default=3, help="Maximum number of recycles [3]")
    parser.add_argument("-use_trf", action='store_true', default=False, help="Do local optimization using trFold")
    parser.add_argument("-use_fp32", action='store_true', default=False, help="Use single precision float")
    parser.add_argument("-write_pdb", action='store_true', default=False, help="Write PDB")
    args = parser.parse_args()
    return args

args = get_args()
ptype = args.p

class Predictor():
    def __init__(self, maxlat=128, maxseq=1024, maxcycle=3, model_dir=None, device=ptype, write_pdb_output=False):
        if model_dir == None:
            self.model_dir = "%s/models"%(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.model_dir = model_dir
        #
        # define model name
        self.model_name = "BFF"
        self.device = device
        self.active_fn = nn.Softmax(dim=1)
        self.write_pdb_output = write_pdb_output
        
        self.params = {'MAXLAT': maxlat, 'MAXSEQ': maxseq, 'MAXCYCLE': maxcycle}

        # define model & load model
        self.model = RoseTTAFoldModule(**MODEL_PARAM).to(self.device)
        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

    def load_model(self, model_name, suffix='last'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True
    
    def predict(self, a3m_fn, out_prefix, L1, n_latent=128, use_trf=False, use_fp32=False):
        msa_orig, ins_orig = parse_a3m(a3m_fn)
        N, L = msa_orig.shape
        print ("INFO: number of sequences & residues are %d, %d"%(N, L))
        
        xyz_t = torch.full((1,L,3,3),np.nan).float()
        t1d = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,L,1)).float()), -1)
        
        # template features
        xyz_t = xyz_t.float().unsqueeze(0)
        t1d = t1d.float().unsqueeze(0)
        t2d = xyz_to_t2d(xyz_t)
        xyz_t = get_init_xyz(xyz_t) # initialize coordinates with first template
        #
        self.model.eval()
        for i_trial in range(1):
            if os.path.exists("%s_%02d.pdb"%(out_prefix, i_trial)):
                continue
            self.run_prediction(msa_orig, ins_orig, t1d, t2d, xyz_t, xyz_t[:,0], t1d[:,0], "%s_%02d"%(out_prefix, i_trial), L1, n_latent=n_latent, use_trf=use_trf, use_fp32=use_fp32)
            torch.cuda.empty_cache()

    def run_prediction(self, msa_orig, ins_orig, t1d, t2d, xyz_t, xyz, state, out_prefix, L1, n_latent=128, use_trf=False, use_fp32=False):
        USE_AMP = not use_fp32
        N, L = msa_orig.shape
        with torch.no_grad():
            if msa_orig.shape[0] > 4096:
                msa, ins = MSABlockDeletion(msa_orig, ins_orig)
                msa = torch.tensor(msa).long().to(self.device) # (N, L)
                ins = torch.tensor(ins).long().to(self.device)
            else:
                msa = torch.tensor(msa_orig).long().to(self.device) # (N, L)
                ins = torch.tensor(ins_orig).long().to(self.device)
            #
            N, L = msa.shape
            seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params=self.params, p_mask=-1.0)
            #
            idx_pdb = torch.arange(L).long().view(1, L)
            idx_pdb[:,L1:] += 200
            #
            seq = seq.unsqueeze(0)
            msa_seed = msa_seed.unsqueeze(0)
            msa_extra = msa_extra.unsqueeze(0)
            t1d = t1d.to(self.device)
            t2d = t2d.to(self.device)
            idx_pdb = idx_pdb.to(self.device)
            xyz_t = xyz_t.to(self.device)
            xyz = xyz.to(self.device)
            state = state.to(self.device)
            
            msa_prev=None
            pair_prev=None
            xyz_prev=xyz
            best_lddt = torch.tensor([-1.0], device=seq.device)
            best_xyz = None
            best_logit = None
            best_aa = None
            for i_cycle in range(self.params['MAXCYCLE']):
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logit_s, logit_aa_s, init_crds, pred_lddt, msa_prev, pair_prev = self.model(msa_seed[:,i_cycle], msa_extra[:,i_cycle],
                                                               seq[:,i_cycle], xyz_prev, 
                                                               state,
                                                               idx_pdb,
                                                               t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                               msa_prev=msa_prev,
                                                               pair_prev=pair_prev)
                    state = torch.cat((nn.Softmax(dim=-1)(logit_aa_s), pred_lddt.unsqueeze(-1)), dim=-1)
                xyz_prev = init_crds
            
                print ("RECYCLE", i_cycle, pred_lddt.mean(), best_lddt.mean())
                best_xyz = init_crds.clone()
                best_logit = logit_s
                best_aa = logit_aa_s
                best_lddt = pred_lddt.clone()
        
        prob_s = list()
        for logit in best_logit:
            prob = self.active_fn(logit.float()) # distogram
            prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
            prob_s.append(prob)
        
        for prob in prob_s:
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
        
        if self.write_pdb_output:
            self.write_pdb(seq[0, -1], best_xyz[0], L1, Bfacts=best_lddt[0], prefix="%s_init"%(out_prefix))
            if use_trf:
                xyz = best_xyz[0, :, 1] # initial ca coordinates
                TRF = TRFold(prob_s, fold_params)
                xyz = TRF.fold(xyz, batch=45, lr=0.1, nsteps=200)
                self.write_pdb(seq[0, -1], xyz, L1, prefix="%s"%(out_prefix), Bfacts=pred_lddt[0])

        prob = np.sum(prob_s[0].permute(1,2,0).detach().cpu().numpy()[:,:]).astype(np.float16)
        np.savez_compressed("%s.npz"%(out_prefix), dist=prob, plddt=best_lddt[0].cpu().numpy())
#        print ("Final results: %.4f %.4f"%(prob.max(), prob[:-10,10:].max()))

                    
    def write_pdb(self, seq, atoms, L1, Bfacts=None, prefix=None):
        L = len(seq)
        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp( Bfacts, 0, 1)
            
            for i,s in enumerate(seq):
                if (len(atoms.shape)==2):
                    if i+1 > L1:
                        chain = "B"
                        resNo = i+1 - L1
                    else:
                        chain = "A"
                        resNo = i+1
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, " CA ", util.num2aa[s], 
                             chain, resNo, atoms[i,0], atoms[i,1], atoms[i,2],
                            1.0, Bfacts[i] ) )
                    if i+1 == L1:
                        f.write("TER\n")
                    ctr += 1

                elif atoms.shape[1]==3:
                    if i+1 > L1:
                        chain = "B"
                        resNo = i+1 - L1
                    else:
                        chain = "A"
                        resNo = i+1
                    for j,atm_j in enumerate((" N  "," CA "," C  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                chain, resNo, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1                
                    if i+1 == L1:
                        f.write("TER\n")
        
if __name__ == "__main__":
    args = get_args()
    pred = Predictor(maxlat=args.maxlat, maxseq=args.maxseq, maxcycle=args.maxcycle, write_pdb_output=args.write_pdb)
    
    tar_s = [line.split() for line in open(args.list)]
    for tar_info in tar_s:
        msa_fn = tar_info[0]
        out_prefix = tar_info[-1]
        if not os.path.exists(msa_fn):
            continue
        if os.path.exists("%s_00.npz"%out_prefix):
            continue
        dir_name = "/".join(out_prefix.split('/')[:-1])
        if dir_name != "":
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        L_s = [int(L) for L in tar_info[1:-1]]
        print ("running %s"%out_prefix)
        start = time.time()
        pred.predict(msa_fn, out_prefix, L_s[0], use_trf=args.use_trf, use_fp32=args.use_fp32)
        print ("Done", time.time()-start)
