import sys, os
import time
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
from data_loader import get_train_valid_set, loader_tbm, loader_fb, Dataset, DistilledDataset, DistributedWeightedSampler
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, xyz_to_bbtor, get_init_xyz
from RoseTTAFoldModel  import RoseTTAFoldModule
from loss import *
#from scheduler import get_linear_schedule_with_warmup, CosineAnnealingWarmupRestarts
from scheduler import get_linear_schedule_with_warmup, get_stepwise_decay_schedule_with_warmup

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
#torch.autograd.set_detect_anomaly(True)
random_seed = 596
torch.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
## To reproduce errors
#import random
np.random.seed(5759)
#random.seed(0)

USE_AMP = False
torch.set_num_threads(4)
MAX_CYCLE = 8
#RECYCLE_WEIGHTS = torch.tensor([max(i, 4) for i in range(1, MAX_CYCLE+1)]).float()
#RECYCLE_WEIGHTS = 1.0 / RECYCLE_WEIGHTS


N_PRINT_TRAIN = 64
#BATCH_SIZE = 1 * torch.cuda.device_count()

LOAD_PARAM = {'shuffle': False,
              'num_workers': 4,
              'pin_memory': True}

def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)

class Trainer():
    def __init__(self, model_name='BFF',
                 n_epoch=100, step_lr=100, lr=1.0e-4, l2_coeff=1.0e-2, port=None, interactive=False,
                 model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1):
        self.model_name = "BFF"
        #self.model_name = "%s_%d_%d_%d_%d"%(model_name, model_param['n_module'], 
        #                                    model_param['n_module_str'],
        #                                    model_param['d_msa'],
        #                                    model_param['d_pair'])
        #
        self.n_epoch = n_epoch
        self.step_lr = step_lr
        self.init_lr = lr
        self.l2_coeff = l2_coeff
        self.port = port
        self.interactive = interactive
        #
        self.model_param = model_param
        self.loader_param = loader_param
        self.loss_param = loss_param
        self.ACCUM_STEP = accum_step
        self.batch_size = batch_size
        print (self.model_param)
        print (self.loader_param)
        print (self.loss_param)

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)
        
    def calc_loss(self, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s,
                  pred, true, pred_lddt, idx, unclamp=False,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, 
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0):
        B, L = true.shape[:2]

        loss_s = list()
        tot_loss = 0.0
       
        # c6d loss
        for i in range(4):
            loss = self.loss_fn(logit_s[i], label_s[...,i]).mean()
            tot_loss += w_dist*loss
            loss_s.append(loss[None].detach())

        # masked token prediction loss
        loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / (mask_aa_s.sum() + 1e-8)
        tot_loss += w_aa*loss
        loss_s.append(loss[None].detach())
        
        # Structural loss
        if unclamp:
            tot_str, str_loss = calc_str_loss(pred, true, A=20.0, gamma=0.9)
        else:
            tot_str, str_loss = calc_str_loss(pred, true, A=10.0, gamma=0.9)
        tot_loss += w_str*tot_str
        loss_s.append(str_loss)
        
        lddt_loss, ca_lddt = calc_lddt_loss(pred[:,:,:,1], true[:,:,1], pred_lddt, idx)
        tot_loss += w_lddt*lddt_loss
        loss_s.append(lddt_loss.detach()[None])
        loss_s.append(ca_lddt.detach())
        
        # bond geometry
        blen_loss, bang_loss = calc_BB_bond_geom(pred[-1], true)
        if w_blen > 0.0:
            tot_loss += w_blen*blen_loss
        if w_bang > 0.0:
            tot_loss += w_bang*bang_loss

        # pseudo dihedral
        dih_loss = calc_pseudo_dih(pred[:,:,:,1], true[:,:,1])
        tot_loss += dih_loss

        loss_s.append(torch.stack((dih_loss, blen_loss, bang_loss)).detach())
        
        return tot_loss, torch.cat(loss_s, dim=0)

    def calc_acc(self, prob, dist, idx_pdb):
        B = idx_pdb.shape[0]
        L = idx_pdb.shape[1] # (B, L)
        seqsep = torch.abs(idx_pdb[:,:,None] - idx_pdb[:,None,:]) + 1
        mask = seqsep > 24
        mask = torch.triu(mask.float())
        #
        cnt_ref = dist < 12
        cnt_ref = cnt_ref.float() * mask
        #
        cnt_pred = prob[:,:12,:,:].sum(dim=1) * mask
        #
        top_pred = torch.topk(cnt_pred.view(B,-1), L)
        kth = top_pred.values.min(dim=-1).values
        tmp_pred = list()
        for i_batch in range(B):
            tmp_pred.append(cnt_pred[i_batch] > kth[i_batch])
        cnt_pred = torch.stack(tmp_pred, dim=0)
        cnt_pred = cnt_pred.float()*mask
        #
        condition = torch.logical_and(cnt_pred==cnt_ref, cnt_ref==torch.ones_like(cnt_ref))
        n_good = condition.float().sum()
        n_total = (cnt_ref == torch.ones_like(cnt_ref)).float().sum() + 1e-9
        n_total_pred = (cnt_pred == torch.ones_like(cnt_pred)).float().sum() + 1e-9
        prec = n_good / n_total_pred
        recall = n_good / n_total
        F1 = 2.0*prec*recall / (prec+recall+1e-9)
        return torch.stack([prec, recall, F1])

    def load_model(self, model, optimizer, scheduler, scaler, model_name, rank, suffix='last', resume_train=False):
        chk_fn = "models/%s_%s.pt"%(model_name, suffix)
        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            return -1, best_valid_loss
        map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        rename_model = False
        for param in model.module.model.state_dict():
            if param not in checkpoint['model_state_dict']:
                rename_model=True
                break
        new_chk = checkpoint['model_state_dict']
        model.module.model.load_state_dict(new_chk, strict=False)
        model.module.shadow.load_state_dict(new_chk, strict=False)
        if resume_train and (not rename_model):
            loaded_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler.last_epoch = loaded_epoch + 1
            #if 'best_loss' in checkpoint:
            #    best_valid_loss = checkpoint['best_loss']
        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description):
        if not os.path.exists("models"):
            os.mkdir("models")
        name = "%s_%s.pt"%(model_name, description)
        return os.path.join("models", name)
    
    # main entry function of training
    # 1) make sure ddp env vars set
    # 2) figure out if we launched using slurm or interactively
    #   - if slurm, assume 1 job launched per GPU
    #   - if interactive, launch one job for each GPU on node
    def run_model_training(self, world_size):
        if ('MASTER_ADDR' not in os.environ):
            os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
        if ('MASTER_PORT' not in os.environ):
            os.environ['MASTER_PORT'] = '%d'%self.port

        if (not self.interactive and "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
            print ("Launched from slurm", rank, world_size)
            self.train_model(rank, world_size)
        else:
            print ("Launched from interactive")
            world_size = torch.cuda.device_count()
            mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

    def train_model(self, rank, world_size):
        #print ("running ddp on rank %d, world_size %d"%(rank, world_size))
        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        #define dataset & data loader
        pdb_IDs, pdb_weights, pdb_dict, fb_IDs, fb_weights, fb_dict, valid_dict = get_train_valid_set(self.loader_param)
        self.n_train = 112*50*4
        #self.n_train = len(pdb_IDs)
        self.n_valid = len(valid_dict.keys())
        self.n_valid = (self.n_valid // world_size)*world_size
        
        train_set = DistilledDataset(pdb_IDs, loader_tbm, pdb_dict,
                                     fb_IDs, loader_fb, fb_dict,
                                     self.loader_param)
        #train_set = Dataset(pdb_IDs, loader_tbm, pdb_dict, self.loader_param)
        valid_set = Dataset(list(valid_dict.keys())[:self.n_valid], loader_tbm, valid_dict, self.loader_param)
        #
        train_sampler = DistributedWeightedSampler(train_set, pdb_weights, fb_weights, 
                                                   num_pdb_per_epoch=112*50,
                                                   num_replicas=world_size, rank=rank)
        #train_sampler = data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=True)
        valid_sampler = data.distributed.DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
       
        train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, **LOAD_PARAM)
        valid_loader = data.DataLoader(valid_set, sampler=valid_sampler, **LOAD_PARAM)
        
        # define model
        model = EMA(RoseTTAFoldModule(**self.model_param).to(gpu), 0.999)
        #for n, p in model.named_parameters():
        #    if "refine_net" not in n:
        #        if "extractor.norm_state" in n:
        #            continue
        #        if "extractor.pred_lddt" in n:
        #            continue
        #        p.requires_grad = False

        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        if rank == 0:
            print ("# of parameters:", count_parameters(ddp_model))
        
        # define optimizer and scheduler
        opt_params = add_weight_decay(ddp_model, self.l2_coeff)
        #optimizer = torch.optim.Adam(opt_params, lr=self.init_lr)
        optimizer = torch.optim.AdamW(opt_params, lr=self.init_lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.step_lr, gamma=0.5)
        #scheduler = get_linear_schedule_with_warmup(optimizer, 1000, 500000)
        #scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 1000, 5000, 0.95)
        scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 5000, 0.95)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
       
        # load model
        loaded_epoch, best_valid_loss = self.load_model(ddp_model, optimizer, scheduler, scaler, 
                                                        self.model_name, gpu, resume_train=False)
        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return
        for epoch in range(loaded_epoch+1, self.n_epoch):
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
          
            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch)
            
            valid_tot, valid_loss, valid_acc = self.valid_cycle(ddp_model, valid_loader, rank, gpu, world_size, epoch)

            if rank == 0: # save model
                if valid_tot < best_valid_loss:
                    best_valid_loss = valid_tot
                    torch.save({'epoch': epoch,
                                #'model_state_dict': ddp_model.state_dict(),
                                'model_state_dict': ddp_model.module.shadow.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'best_loss': best_valid_loss,
                                'train_loss': train_loss,
                                'train_acc': train_acc,
                                'valid_loss': valid_loss,
                                'valid_acc': valid_acc},
                                self.checkpoint_fn(self.model_name, 'best'))
            
            
                torch.save({'epoch': epoch,
                            #'model_state_dict': ddp_model.state_dict(),
                            'model_state_dict': ddp_model.module.shadow.state_dict(),
                            'final_state_dict': ddp_model.module.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'valid_loss': valid_loss,
                            'valid_acc': valid_acc,
                            'best_loss': best_valid_loss},
                            self.checkpoint_fn(self.model_name, 'last'))
                
        dist.destroy_process_group()

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch):
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        
        # For intermediate logs
        local_tot = 0.0
        local_loss = None
        local_acc = None
        train_tot = 0.0
        train_loss = None
        train_acc = None

        counter = 0
        
        #g = torch.Generator()
        #g.manual_seed(epoch+random_seed)
        
        for seq, msa, msa_masked, msa_full, mask, true_crds, idx_pdb, xyz_t, t1d, unclamp in train_loader:
            # transfer inputs to device
            B, _, N, L = msa.shape
            
            idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
            true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 3, 3)

            xyz_t = xyz_t.to(gpu, non_blocking=True)
            t1d = t1d.to(gpu, non_blocking=True)

            seq = seq.to(gpu, non_blocking=True)
            msa = msa.to(gpu, non_blocking=True)
            msa_masked = msa_masked.to(gpu, non_blocking=True)
            msa_full = msa_full.to(gpu, non_blocking=True)
            mask = mask.to(gpu, non_blocking=True)
            
            # processing labels & template features
            c6d, _ = xyz_to_c6d(true_crds)
            c6d = c6d_to_bins2(c6d)
            t2d = xyz_to_t2d(xyz_t)
            xyz_t = get_init_xyz(xyz_t)
            xyz_prev = xyz_t[:,0]
            state = t1d[:,0]

            counter += 1

            N_cycle = np.random.randint(1, MAX_CYCLE+1) # number of recycling
            #sampled = torch.multinomial(RECYCLE_WEIGHTS, 1, False, generator=g)[0]
            #N_cycle = int(sampled) + 1
            
            msa_prev = None
            pair_prev = None
            with torch.no_grad():
                for i_cycle in range(N_cycle-1):
                    with ddp_model.no_sync():
                        with torch.cuda.amp.autocast(enabled=USE_AMP):
                            msa_prev, pair_prev, xyz_prev, logit_aa_s, lddt_prev = ddp_model(msa_masked[:,i_cycle],
                                                                      msa_full[:,i_cycle],
                                                                      seq[:,i_cycle], xyz_prev, state, idx_pdb,
                                                                      t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                                      msa_prev=msa_prev,
                                                                      pair_prev=pair_prev,
                                                                      return_raw=True,
                                                                      use_checkpoint=False)
                            state = torch.cat((nn.Softmax(dim=-1)(logit_aa_s), lddt_prev), dim=-1)

            i_cycle = N_cycle-1

            if counter%self.ACCUM_STEP != 0:
                with ddp_model.no_sync():
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logit_s, logit_aa_s, pred_crds, pred_lddts = ddp_model(msa_masked[:,i_cycle],
                                                                   msa_full[:,i_cycle],
                                                                   seq[:,i_cycle], xyz_prev, state, idx_pdb,
                                                                   t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                                   msa_prev=msa_prev,
                                                                   pair_prev=pair_prev,
                                                                   use_checkpoint=True)
                        prob = self.active_fn(logit_s[0]) # distogram
                        acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb)

                        loss, loss_s = self.calc_loss(logit_s, c6d,
                                logit_aa_s, msa[:, i_cycle], mask[:,i_cycle],
                                pred_crds, true_crds, pred_lddts, idx_pdb, unclamp=unclamp, **self.loss_param)
                    loss = loss / self.ACCUM_STEP
                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logit_s, logit_aa_s, pred_crds, pred_lddts = ddp_model(msa_masked[:,i_cycle],
                                                               msa_full[:,i_cycle],
                                                               seq[:,i_cycle], xyz_prev, state, idx_pdb,
                                                               t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                               msa_prev=msa_prev,
                                                               pair_prev=pair_prev,
                                                               use_checkpoint=True)
                    prob = self.active_fn(logit_s[0]) # distogram
                    acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb)

                    loss, loss_s = self.calc_loss(logit_s, c6d,
                            logit_aa_s, msa[:, i_cycle], mask[:,i_cycle],
                            pred_crds, true_crds, pred_lddts, idx_pdb, unclamp=unclamp, **self.loss_param)
                loss = loss / self.ACCUM_STEP
                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale != scaler.get_scale())
                optimizer.zero_grad()
                if not skip_lr_sched:
                    scheduler.step()
                ddp_model.module.update() # apply EMA
            
            ## check parameters with no grad
            #if rank == 0:
            #    for n, p in ddp_model.named_parameters():
            #        if p.grad is None and p.requires_grad is True:
            #            print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model
            

            local_tot += loss.detach()*self.ACCUM_STEP
            if local_loss == None:
                local_loss = torch.zeros_like(loss_s.detach())
                local_acc = torch.zeros_like(acc_s.detach())
            local_loss += loss_s.detach()
            local_acc += acc_s.detach()
            
            train_tot += loss.detach()*self.ACCUM_STEP
            if train_loss == None:
                train_loss = torch.zeros_like(loss_s.detach())
                train_acc = torch.zeros_like(acc_s.detach())
            train_loss += loss_s.detach()
            train_acc += acc_s.detach()

            
            if counter % N_PRINT_TRAIN == 0:
                if rank == 0:
                    max_mem = torch.cuda.max_memory_allocated()/1e9
                    train_time = time.time() - start_time
                    local_tot /= float(N_PRINT_TRAIN)
                    local_loss /= float(N_PRINT_TRAIN)
                    local_acc /= float(N_PRINT_TRAIN)
                    
                    local_tot = local_tot.cpu().detach()
                    local_loss = local_loss.cpu().detach().numpy()
                    local_acc = local_acc.cpu().detach().numpy()

                    sys.stdout.write("Local: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | Max mem %.4f\n"%(\
                            epoch, self.n_epoch, counter*self.batch_size*world_size, self.n_train, train_time, local_tot, \
                            " ".join(["%8.4f"%l for l in local_loss]),\
                            local_acc[0], local_acc[1], local_acc[2], max_mem))
                    sys.stdout.flush()
                    local_tot = 0.0
                    local_loss = None 
                    local_acc = None 
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # write total train loss
        train_tot /= float(counter * world_size)
        train_loss /= float(counter * world_size)
        train_acc  /= float(counter * world_size)

        dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        train_tot = train_tot.cpu().detach()
        train_loss = train_loss.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.n_train, self.n_train, train_time, train_tot, \
                    " ".join(["%8.4f"%l for l in train_loss]),\
                    train_acc[0], train_acc[1], train_acc[2]))
            sys.stdout.flush()
            
        return train_tot, train_loss, train_acc

    def valid_cycle(self, ddp_model, valid_loader, rank, gpu, world_size, epoch):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_lddt = None
        counter = 0
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask, true_crds, idx_pdb, xyz_t, t1d, unclamp in valid_loader:
                # transfer inputs to device
                B, _, N, L = msa.shape
                
                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 3, 3)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask = mask.to(gpu, non_blocking=True)
                
                # processing labels & template features
                c6d, _ = xyz_to_c6d(true_crds)
                c6d = c6d_to_bins2(c6d)
                t2d = xyz_to_t2d(xyz_t)
                xyz_t = get_init_xyz(xyz_t)
                xyz_prev = xyz_t[:,0]
                state = t1d[:,0]
            
                # set number of recycles
                N_cycle = MAX_CYCLE
                msa_prev = None
                pair_prev = None
                lddt_s = list()
                for i_cycle in range(N_cycle-1): 
                    
                    msa_prev, pair_prev, xyz_prev, logit_aa_s, lddt_prev = ddp_model(msa_masked[:,i_cycle],
                                                              msa_full[:,i_cycle],
                                                              seq[:,i_cycle], xyz_prev, state, idx_pdb,
                                                              t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                              msa_prev=msa_prev,
                                                              pair_prev=pair_prev,
                                                              return_raw=True,
                                                              use_checkpoint=False)
                    state = torch.cat((nn.Softmax(dim=-1)(logit_aa_s), lddt_prev), dim=-1)
                    lddt = calc_lddt(xyz_prev[:,:,1,:][None], true_crds[:,:,1,:])
                    lddt_s.append(lddt.detach())
                    
                    # TODO: get a structure quality in terms of CA-lddt?
                i_cycle = N_cycle-1
                logit_s, logit_aa_s, pred_crds, pred_lddts = ddp_model(msa_masked[:,i_cycle],
                                                              msa_full[:,i_cycle],
                                                              seq[:,i_cycle], xyz_prev, state, idx_pdb,
                                                              t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                              msa_prev=msa_prev,
                                                              pair_prev=pair_prev,
                                                              use_checkpoint=False)
                prob = self.active_fn(logit_s[0]) # distogram
                acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb)

                loss, loss_s = self.calc_loss(logit_s, c6d,
                        logit_aa_s, msa[:,i_cycle], mask[:,i_cycle],
                        pred_crds, true_crds, pred_lddts, idx_pdb, unclamp=unclamp, **self.loss_param)
                lddt = calc_lddt(pred_crds[-1:,:,:,1,:], true_crds[:,:,1,:])
                lddt_s.append(lddt.detach())
                lddt_s = torch.cat(lddt_s)
                
                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_lddt = torch.zeros_like(lddt_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_lddt += lddt_s.detach()
                counter += 1

            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        valid_lddt /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_lddt, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_lddt = valid_lddt.cpu().detach().numpy()
        
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Valid: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.n_valid, self.n_valid, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    " ".join(["%8.4f"%l for l in valid_lddt]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc
    

if __name__ == "__main__":
    from arguments import get_args
    args, model_param, loader_param, loss_param = get_args()

    mp.freeze_support()
    train = Trainer(model_name=args.model_name,
                    n_epoch=args.num_epochs, step_lr=args.step_lr, lr=args.lr, l2_coeff=1.0e-2,
                    port=args.port, model_param=model_param, loader_param=loader_param, 
                    loss_param=loss_param, 
                    batch_size=args.batch_size,
                    accum_step=args.accum)
    train.run_model_training(torch.cuda.device_count())
