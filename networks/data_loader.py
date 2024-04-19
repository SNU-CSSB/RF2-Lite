import torch
from torch.utils import data
import os
import csv
from dateutil import parser
import numpy as np
from parsers import parse_a3m, parse_pdb

# training on blue
base_dir = "/gscratch2/PDB-2021AUG02"
compl_dir = "/gscratch2/RoseTTAComplex"
fb_dir = "/gscratch2/fb_af1"

def set_data_loader_params(args):
    PARAMS = {
        "COMPL_LIST" : "%s/list.complex.csv"%compl_dir,
        "NEGATIVE_LIST" : "%s/list.negative.csv"%compl_dir,
        #"PDB_LIST"   : "%s/list_v01.csv"%base_dir,
        "PDB_LIST"   : "%s/list_v01.csv"%compl_dir,
        "FB_LIST"    : "%s/list_b1-3.csv"%fb_dir,
        "VAL_COMPL"  : "%s/val_lists/xaa"%compl_dir,
        "VAL_NEG"    : "%s/val_lists/xaa.neg"%compl_dir,
        "PDB_DIR"    : base_dir,
        "FB_DIR"     : fb_dir,
        "COMPL_DIR"  : compl_dir,
        "MINSEQ"     : 1,
        "MAXSEQ"     : 1024,
        "MAXLAT"     : 128, 
        "CROP"       : 256,
        "DATCUT"     : "2030-Jan-01",
        "RESCUT"     : 3.5,
        "PLDDTCUT"   : 70.0,
        "SLICE"      : "DISCONT",
        "ROWS"       : 1,
        "BLOCKCUT"   : 10,
        "MAXCYCLE"   : 4
    }
    for param in PARAMS:
        if hasattr(args, param.lower()):
            PARAMS[param] = getattr(args, param.lower())
    return PARAMS

def MSABlockDeletion(msa, ins, nb=5):
    '''
    Input: MSA having shape (N, L)
    output: new MSA with block deletion
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, np.bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    csum = torch.zeros(N_seq, N_res, data.shape[-1], device=data.device).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def MSAFeaturize(msa, ins, params={'MAXLAT': 128, 'MAXSEQ': 1024, 'MAXCYCLE': 3}, eps=1e-6, p_mask=0.15):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
    '''
    N, L = msa.shape
        
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0) 

    # Select Nclust sequence randomly (seed MSA or latent MSA)
    Nclust = min(N, params['MAXLAT'])
    #nmin = min(N, params['MINSEQ'])
    #Nclust = np.random.randint(nmin, max(nmin, min(N, params['MAXLAT']))+1)
    Nextra = N-Nclust
    if Nextra < 1:
        Nextra = 1
    Nextra = min(Nextra, params['MAXSEQ'])
    #Nextra = min(Nextra, params['MAXTOKEN'] // L)
    #Nextra = min(Nextra, params['MAXSEQ'])
    #Nextra = np.random.randint(1, Nextra+1)
    #
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        sample = torch.randperm(N-1, device=msa.device)
        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking 
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05]*20 + [0.0]], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=21)
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
        
        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
        b_seq.append(msa_masked[0].clone())
        
        # get extra sequenes
        if Nclust < N: # there are extra sequences
            msa_extra = msa[1:,:][sample[Nclust-1:]]
            ins_extra = ins[1:,:][sample[Nclust-1:]]
            extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
        else:
            msa_extra = msa_masked[:1]
            ins_extra = ins[:1]
            extra_mask = mask_pos[:1]
        N_extra = msa_extra.shape[0]
        
        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=22)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=22)
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20).float() # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(~extra_mask, msa_extra != 20).float() 
        agreement = torch.matmul((count_extra[:,:,None]*msa_extra_onehot).view(N_extra, -1), (count_clust[:,:,None]*msa_clust_onehot).view(Nclust, -1).T)
        assignment = torch.argmax(agreement, dim=-1)

        # seed MSA features
        # 1. one_hot encoded aatype: msa_clust_onehot
        # 2. cluster profile
        count_extra = ~extra_mask
        count_clust = ~mask_pos
        msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:,:,None]*msa_clust_profile
        count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L)
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:,:,None]
        # 3. insertion statistics
        msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
        msa_clust_del += count_clust*ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
        msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
        #
        msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust), dim=-1)

        # extra MSA features
        ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
        msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:Nextra][:,:,None]), dim=-1)

        b_msa_clust.append(msa_clust)
        b_msa_seed.append(msa_seed)
        b_msa_extra.append(msa_extra)
        b_mask_pos.append(mask_pos)
    
    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def TemplFeaturize(tplt, qlen, params, pick_top=False):
    seqID_cut = params['seqID']
    if seqID_cut <= 100.0:
        sel = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[sel]
        tplt['qmap'] = tplt['qmap'][:,sel]
        tplt['xyz'] = tplt['xyz'][:, sel]
        tplt['seq'] = tplt['seq'][:, sel]
        tplt['f1d'] = tplt['f1d'][:, sel]

    ntplt = len(tplt['ids'])
    if ntplt<1: # no templates
        xyz = torch.full((1,qlen,3,3),np.nan).float()
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,qlen,1)).float()), -1)
        return xyz, t1d
   
    npick = np.random.randint(params['MINTPLT'], min(ntplt, params['MAXTPLT'])+1)
    if not pick_top:
        sample = torch.randperm(ntplt)[:npick]
    else:
        sample = torch.arange(npick)

    xyz = torch.full((npick,qlen,3,3),np.nan).float()
    t1d = torch.full((npick, qlen), 20).long()
    t1d_val = torch.zeros((npick, qlen, 1)).float()

    for i,nt in enumerate(sample):
        sel = torch.where(tplt['qmap'][0,:,1]==nt)[0]
        pos = tplt['qmap'][0,sel,0]
        xyz[i,pos] = tplt['xyz'][0,sel,:3]
        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2].unsqueeze(-1) # alignment confidence

    t1d = torch.nn.functional.one_hot(t1d, num_classes=21).float()
    t1d = torch.cat((t1d, t1d_val), dim=-1)

    return xyz, t1d

def get_train_valid_set(params):
    # get sequence hash - cluster information for all monomers
    h2c = {}
    with open(params['PDB_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            h2c[r[3]] = int(r[4])

    # read validation IDs
    OFFSET = 100000 # to avoid ovelap of cluster id between positive / negative sets
    val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])
    val_neg_ids = set([int(l)+OFFSET for l in open(params['VAL_NEG']).readlines()])
    val_clust = list()

    # read & clean list.complex.csv
    with open(params['COMPL_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy information
        rows = [[r[0],r[3],int(r[4]),[int(plen) for plen in r[5].split(':')],r[6]] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]

    # compile training and validation sets
    rows_subset = rows[::params['ROWS']]
    train = {}
    valid = {}
    for r in rows_subset:
        if r[2] in val_compl_ids:
            hash_A, hash_B = r[1].split('_')
            val_clust.append(h2c[hash_A])
            val_clust.append(h2c[hash_B])
            if r[2] in valid.keys():
                valid[r[2]].append((r[:2], r[-2], r[-1], False)) # ((pdb,hash), length, taxonomy, negative?)
            else:
                valid[r[2]] = [(r[:2], r[-2], r[-1], False)]
        else:
            if r[2] in train.keys():
                train[r[2]].append((r[:2], r[-2], r[-1]))
            else:
                train[r[2]] = [(r[:2], r[-2], r[-1])]
    
    # compile negative examples
    neg = {}
    with open(params['NEGATIVE_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy information
        rows = [[r[0],r[3],OFFSET+int(r[4]),[int(plen) for plen in r[5].split(':')],r[6]] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    for r in rows:
        if r[2] in val_neg_ids:
            hash_A, hash_B = r[1].split('_')
            val_clust.append(h2c[hash_A])
            val_clust.append(h2c[hash_B])
            if r[2] in valid.keys():
                valid[r[2]].append((r[:2], r[-2], r[-1], True)) # ((pdb,hash), length, taxonomy, negative?)
            else:
                valid[r[2]] = [(r[:2], r[-2], r[-1], True)]
        else:
            if r[2] in neg.keys():
                neg[r[2]].append((r[:2], r[-2], r[-1])) # ((pdb,hash), length, taxonomy)
            else:
                neg[r[2]] = [(r[:2], r[-2], r[-1])]

    val_clust = set(val_clust)

    # compile monomer PDB sets
    # remove pdbs included in validation set
    pdb = {}
    with open(params['PDB_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # read pdb_chain, hash, cluster, length
        rows = [[r[0],r[3],int(r[4]),len(r[-1])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT']) and
                int(r[4]) not in val_clust]
    for r in rows:
        if r[2] in pdb.keys():
            pdb[r[2]].append((r[:2], r[-1]))
        else:
            pdb[r[2]] = [(r[:2], r[-1])]

    # compile facebook model sets
    with open(params['FB_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[2],int(r[3]),len(r[-1].strip())] for r in reader
                 if float(r[1]) > params['PLDDTCUT'] and
                 len(r[-1].strip()) > 200]
    fb = {}
    for r in rows:
        if r[2] in fb.keys():
            fb[r[2]].append((r[:2], r[-1]))
        else:
            fb[r[2]] = [(r[:2], r[-1])]
    
    # Get average chain length in each cluster and calculate weights
    train_IDs = list(train.keys())
    neg_IDs = list(neg.keys())
    pdb_IDs = list(pdb.keys())
    fb_IDs = list(fb.keys())
    train_weights = list()
    neg_weights = list()
    pdb_weights = list()
    fb_weights = list()
    for key in train_IDs:
        plen = sum([sum(plen) for _, plen, _ in train[key]]) // len(train[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        train_weights.append(w)
    
    for key in neg_IDs:
        plen = sum([sum(plen) for _, plen, _ in neg[key]]) // len(neg[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        neg_weights.append(w)
    
    for key in pdb_IDs:
        plen = sum([plen for _, plen in pdb[key]]) // len(pdb[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        pdb_weights.append(w)
    
    for key in fb_IDs:
        plen = sum([plen for _, plen in fb[key]]) // len(fb[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        fb_weights.append(w)

    return train_IDs, torch.tensor(train_weights).float(), train, \
           neg_IDs, torch.tensor(neg_weights).float(), neg, \
           pdb_IDs, torch.tensor(pdb_weights).float(), pdb, \
           fb_IDs, torch.tensor(fb_weights).float(), fb, valid

# slice long chains
def get_crop(l, device, params, unclamp=False):

    sel = torch.arange(l,device=device)
    if l < params['CROP']:
        return sel
    
    size = params['CROP']
    
    if params['SLICE'] == 'CONT' or unclamp:
        # slice continuously
        start = np.random.randint(l-size+1)
        return sel[start:start+size]

    elif params['SLICE'] == 'DISCONT':
        # slice discontinuously
        size1 = np.random.randint(size//4, size//2+1)
        size2 = size - size1
        gap = np.random.randint(l-size1-size2+1)
        
        start1 = np.random.randint(l-size1-size2-gap+1)
        sel1 = sel[start1:start1+size1]
        
        start2 = start1 + size1 + gap
        sel2 = sel[start2:start2+size2]
        
        return torch.cat([sel1,sel2])

    else:
        sys.exit('Error: wrong cropping mode:', params['SLICE'])

def get_complex_crop(len_s, device, params):
    tot_len = sum(len_s)
    sel = torch.arange(tot_len, device=device)
    
    n_added = 0
    n_remaining = sum(len_s)
    preset = 0
    sel_s = list()
    for k in range(len(len_s)):
        n_remaining -= len_s[k]
        crop_max = min(params['CROP']-n_added, len_s[k])
        crop_min = min(len_s[k], max(0, params['CROP'] - n_added - n_remaining))
        
        crop_size = np.random.randint(crop_min, crop_max+1)
        n_added += crop_size
        
        start = np.random.randint(0, len_s[k]-crop_size+1) + preset
        sel_s.append(sel[start:start+crop_size])
        preset += len_s[k]
    return torch.cat(sel_s)

def get_spatial_crop(xyz, len_s, params, cutoff=10.0, eps=1e-6):
    tot_len = sum(len_s)
    device = xyz.device
    sel = torch.arange(tot_len, device=device)
    
    # get interface residue
    cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff
    i,j = torch.where(cond)
    ifaces = torch.cat([i,j+len_s[0]])
    if len(ifaces) < 1:
        return get_complex_crop(len_s, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    dist = torch.cdist(xyz[:,1], xyz[cnt_idx,1][None]).reshape(-1) + torch.arange(len(xyz), device=xyz.device)*eps
    _, idx = torch.topk(dist, params['CROP'], largest=False)

    return sel[idx]

def merge_a3m(a3mA, a3mB, L_s):
    # merge msa
    query = torch.cat([a3mA['msa'][0], a3mB['msa'][0]]).unsqueeze(0) # (1, L)
    msa = [query]
    if a3mA['msa'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['msa'][1:], (0,L_s[1]), "constant", 20) # pad gaps
        msa.append(extra_A)
    if a3mB['msa'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['msa'][1:], (L_s[0],0), "constant", 20)
        msa.append(extra_B)
    msa = torch.cat(msa, dim=0)
    
    # merge ins
    query = torch.cat([a3mA['ins'][0], a3mB['ins'][0]]).unsqueeze(0) # (1, L)
    ins = [query]
    if a3mA['ins'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['ins'][1:], (0,L_s[1]), "constant", 0) # pad gaps
        ins.append(extra_A)
    if a3mB['ins'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['ins'][1:], (L_s[0],0), "constant", 0)
        ins.append(extra_B)
    ins = torch.cat(ins, dim=0)
    return {'msa': msa, 'ins': ins}

def loader_complex(item, L_s, taxID, params, iface_biased=False, negative=False):
    pdb_pair = item[0]
    pMSA_hash = item[1]
    
    msaA_id, msaB_id = pMSA_hash.split('_')
    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        if negative:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA.negative/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m'
        else:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m'
        a3m = get_msa(pMSA_fn, pMSA_hash)
    else:
        # read MSA for each subunit & merge them
        a3mA_fn = params['PDB_DIR'] + '/a3m/' + msaA_id[:3] + '/' + msaA_id + '.a3m.gz'
        a3mB_fn = params['PDB_DIR'] + '/a3m/' + msaB_id[:3] + '/' + msaB_id + '.a3m.gz'
        a3mA = get_msa(a3mA_fn, msaA_id)
        a3mB = get_msa(a3mB_fn, msaB_id)
        a3m = merge_a3m(a3mA, a3mB, L_s)

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)

    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbA_id[1:3]+'/'+pdbA_id+'.pt')
    pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbB_id[1:3]+'/'+pdbB_id+'.pt')

    xyz = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)[:,:3] # backbone only
    mask = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)[:,:3].sum(dim=-1)
    mask = ~(mask < 3.0) # whether it has missing BB atoms
    res_idx = torch.arange(sum(L_s))
    res_idx[L_s[0]:] += 200

    chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
    chain_idx[:L_s[0], :L_s[0]] = 1
    chain_idx[L_s[0]:, L_s[0]:] = 1

    # Do cropping
    if sum(L_s) > params['CROP']:
        if iface_biased:
            sel = get_spatial_crop(xyz, L_s, params)
        else:
            sel = get_complex_crop(L_s, xyz.device, params)
        #
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        try:
            xyz = xyz[sel]
        except:
            print ("ERROR", pdb_pair, L_s, xyz.shape, msa_seed.shape, sel)
            sys.exit()
        mask = mask[sel]
        #
        idx = res_idx[sel]
        chain_idx = chain_idx[sel][:,sel]
    else:
        idx = res_idx
    
    xyz = torch.nan_to_num(xyz)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), chain_idx, False, negative

def loader_pdb(item, params, unclamp=False, pick_top=False):
    # TODO: change this to predict disorders rather than ignoring them
    # TODO: add featurization here

    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])
    
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    xyz = pdb['xyz'][:,:3]
    mask = pdb['mask'][:,:3].sum(dim=-1)
    mask = ~(mask < 3.0) # whether it has missing BB atoms

    # Residue cropping
    L = msa.shape[1]
    crop_idx = get_crop(L, msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()

    xyz = torch.nan_to_num(xyz)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, crop_idx.long(), chain_idx, unclamp, False
    
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_fb_pdb(pdbfilename, plddtfilename, item, lddtcut):
    xyz, mask, res_idx = parse_pdb(pdbfilename)
    plddt = np.load(plddtfilename)
    cond = plddt > lddtcut

    mask = np.logical_and(mask, cond[:,None])
    
    return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'label':item}

def get_msa(a3mfilename, item):
    msa,ins = parse_a3m(a3mfilename)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'label':item}

def loader_fb(item, params, unclamp=False):
    # loads sequence/structure/plddt information 
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_fb_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                  item[0], params['PLDDTCUT'])
    
    L = a3m['msa'].shape[-1]

    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    xyz = pdb['xyz'][:,:3]
    mask = pdb['mask'][:,:3].sum(dim=-1)
    mask = ~(mask < 3.0) # whether it has missing BB atoms

    # Residue cropping
    crop_idx = get_crop(L, msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()

    xyz = torch.nan_to_num(xyz)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, crop_idx.long(), chain_idx, unclamp, False

class Dataset(data.Dataset):
    def __init__(self, IDs, loader, compl_dict, params, unclamp_cut=0.9):
        self.IDs = IDs
        self.compl_dict = compl_dict
        self.loader = loader
        self.params = params
        self.unclamp_cut = unclamp_cut
        self.prob = {}
        for ID in self.IDs:
            p = np.array([(1/512.)*max(min(float(sum(p_len)),512.),256.) for info, p_len, taxID, neg in self.compl_dict[ID]])
            p = p / np.sum(p)
            self.prob[ID] = p

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        prob = self.prob[ID]
        sel_idx = np.random.choice(len(self.compl_dict[ID]), p=prob)
        negative = self.compl_dict[ID][sel_idx][-1]
        if negative:
            out = self.loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1],self.compl_dict[ID][sel_idx][2],self.params, iface_biased=False, negative=True)
        else:
            out = self.loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1],self.compl_dict[ID][sel_idx][2],self.params, iface_biased=True)
            #p_unclamp = np.random.rand()
            #if p_unclamp > 0.5:
            #    out = self.loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1],self.compl_dict[ID][sel_idx][2],self.params, iface_biased=True)
            #else:
            #    out = self.loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1],self.compl_dict[ID][sel_idx][2],self.params, iface_biased=False)
        return out

class DistilledDataset(data.Dataset):
    def __init__(self,
                 compl_IDs,
                 compl_loader,
                 compl_dict,
                 neg_IDs,
                 neg_loader,
                 neg_dict,
                 pdb_IDs,
                 pdb_loader,
                 pdb_dict,
                 fb_IDs,
                 fb_loader,
                 fb_dict,
                 params):
        #
        self.compl_IDs = compl_IDs
        self.compl_loader = compl_loader
        self.compl_dict = compl_dict
        self.neg_IDs = neg_IDs
        self.neg_loader = neg_loader
        self.neg_dict = neg_dict
        self.pdb_IDs = pdb_IDs
        self.pdb_dict = pdb_dict
        self.pdb_loader = pdb_loader
        self.fb_IDs = fb_IDs
        self.fb_dict = fb_dict
        self.fb_loader = fb_loader
        self.params = params
        self.unclamp_cut = 0.9
        
        self.compl_inds = np.arange(len(self.compl_IDs))
        self.neg_inds = np.arange(len(self.neg_IDs))
        self.fb_inds = np.arange(len(self.fb_IDs))
        self.pdb_inds = np.arange(len(self.pdb_IDs))
    
    def __len__(self):
        return len(self.fb_inds) + len(self.pdb_inds) + len(self.compl_inds) + len(self.neg_inds)

    def __getitem__(self, index):
        p_unclamp = np.random.rand()
        if index >= len(self.fb_inds) + len(self.pdb_inds) + len(self.compl_inds): # from negative set
            ID = self.neg_IDs[index-len(self.fb_inds)-len(self.pdb_inds)-len(self.compl_inds)]
            sel_idx = np.random.randint(0, len(self.neg_dict[ID]))
            out = self.neg_loader(self.neg_dict[ID][sel_idx][0], self.neg_dict[ID][sel_idx][1], self.neg_dict[ID][sel_idx][2], self.params, iface_biased=False, negative=True)

        elif index >= len(self.fb_inds) + len(self.pdb_inds): # from complex set
            ID = self.compl_IDs[index-len(self.fb_inds)-len(self.pdb_inds)]
            sel_idx = np.random.randint(0, len(self.compl_dict[ID]))
            if p_unclamp > 0.5:
                out = self.compl_loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1],self.compl_dict[ID][sel_idx][2],self.params, iface_biased=True)
            else:
                out = self.compl_loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1],self.compl_dict[ID][sel_idx][2],self.params, iface_biased=False)
        elif index >= len(self.fb_inds): # from PDB set
            ID = self.pdb_IDs[index-len(self.fb_inds)]
            sel_idx = np.random.randint(0, len(self.pdb_dict[ID]))
            if p_unclamp > self.unclamp_cut:
                out = self.pdb_loader(self.pdb_dict[ID][sel_idx][0], self.params, unclamp=True)
            else:
                out = self.pdb_loader(self.pdb_dict[ID][sel_idx][0], self.params, unclamp=False)
        else: # from FB set
            ID = self.fb_IDs[index]
            sel_idx = np.random.randint(0, len(self.fb_dict[ID]))
            if p_unclamp > self.unclamp_cut:
                out = self.fb_loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=True)
            else:
                out = self.fb_loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=False)
        return out

class DistributedWeightedSampler(data.Sampler):
    def __init__(self, dataset, compl_weights, neg_weights, pdb_weights, fb_weights, epoch_unit=6400, num_replicas=None, rank=None, replacement=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        assert epoch_unit % num_replicas == 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_compl_per_epoch = epoch_unit * 2
        self.num_neg_per_epoch = epoch_unit * 2
        self.num_pdb_per_epoch = epoch_unit
        self.num_fb_per_epoch = epoch_unit * 3
        self.total_size = epoch_unit * 8 # (25%: COMPL / 25%: NEGATIVE / 12.5% PDB / 37.5%: FB)
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        self.compl_weights = compl_weights
        self.neg_weights = neg_weights
        self.pdb_weights = pdb_weights
        self.fb_weights = fb_weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (fb + pdb + compl models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # 1. subsample fb, pdb, and compl based on length
        fb_sampled = torch.multinomial(self.fb_weights, self.num_fb_per_epoch, self.replacement, generator=g)
        pdb_sampled = torch.multinomial(self.pdb_weights, self.num_pdb_per_epoch, self.replacement, generator=g)
        compl_sampled = torch.multinomial(self.compl_weights, self.num_compl_per_epoch, self.replacement, generator=g)
        neg_sampled = torch.multinomial(self.neg_weights, self.num_neg_per_epoch, self.replacement, generator=g)
        
        fb_indices = indices[fb_sampled]
        pdb_indices = indices[pdb_sampled + len(self.dataset.fb_IDs)]
        compl_indices = indices[compl_sampled + len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs)]
        neg_indices = indices[neg_sampled + len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs) + len(self.dataset.compl_IDs)]

        indices = torch.cat((fb_indices, pdb_indices, compl_indices, neg_indices))

        # shuffle indices
        indices = indices[torch.randperm(len(indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

