import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import torch.utils.checkpoint as checkpoint
from util_module import *
from Attention_module import *
from SE3_network import SE3TransformerWrapper
from resnet import ResidualNetwork
from constant import *

# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)

# Update MSA with biased self-attention. bias from Pair & Str
class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair+36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.proj_pair = init_lecun_normal(self.proj_pair)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.proj_pair.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair) # (B, L, L, d_pair)
        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.type_as(state)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa += self.drop_row(self.row_attn(msa, pair))
        msa += self.col_attn(msa)
        msa += self.ff(msa)

        return msa

class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super(PairStr2Pair, self).__init__()

        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_pair, d_rbf, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_rbf, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)
        
    def forward(self, pair, rbf_feat):
        pair += self.drop_row(self.row_attn(pair, rbf_feat))
        pair += self.drop_col(self.col_attn(pair, rbf_feat))
        pair += self.ff(pair)
        return pair

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)

        self.proj_down = nn.Linear(d_pair*2, d_pair)
        self.update = ResidualNetwork(1, d_pair, d_pair, d_pair, p_drop=p_drop)
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        self.proj_out = init_lecun_normal(self.proj_out)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)
        nn.init.zeros_(self.proj_out.bias)

        # Identity initialization for proj_down
        nn.init.eye_(self.proj_down.weight)
        nn.init.zeros_(self.proj_down.bias)

    def forward(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right /= float(N)
        out = torch.einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
        
        pair = torch.cat((pair, out), dim=-1) # (B, L, L, d_pair*2)
        pair = self.proj_down(pair)
        pair = self.update(pair.permute(0,3,1,2).contiguous())
        pair = pair.permute(0,2,3,1).contiguous()
        
        return pair

class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
    
        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+36+1, SE3_param['num_edge_features'])
        
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3TransformerWrapper(**SE3_param)
        
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, xyz, state, idx, top_k=128, eps=1e-5):
        # process msa & pair features
        B, N, L = msa.shape[:3]
        msa = self.norm_msa(msa[:,0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)
       
        msa = torch.cat((msa, state), dim=-1)
        msa = self.norm_node(self.embed_x(msa))
        pair = self.norm_edge1(self.embed_e1(pair))
        
        neighbor = get_seqsep(idx)
        ca_xyz = xyz[:,:,1].contiguous()
        rbf_feat = rbf(torch.cdist(ca_xyz, ca_xyz))
        pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))
        
        # define graph
        if top_k != 0:
            G, edge_feats = make_topk_graph(ca_xyz, pair, idx, top_k=top_k)
        else:
            G, edge_feats = make_full_graph(ca_xyz, pair, idx, top_k=top_k)
        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)
        
        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, msa.reshape(B*L, -1, 1), l1_feats, edge_feats)

        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        
        offset = shift['1'].reshape(B, L, 2, 3)
        T = offset[:,:,0,:] / 10.0
        R = offset[:,:,1,:] / 100.0
        R_angle = torch.norm(R, dim=-1, keepdim=True) # (B, L, 1)
        R_vector = R / (R_angle+eps) # (B, L, 3)
        R_vector = R_vector.unsqueeze(-2) # (B, L, 1, 3)
        #
        v = l1_feats.reshape(B, L, -1, 3)
        R_dot_v = (R_vector * v).sum(dim=-1, keepdim=True) # (B, L, 3, 1)
        R_cross_v = torch.cross(R_vector.expand(-1, -1, 3, -1), v, dim=-1) # (B, L, 3, 3)
        v_perpendicular = v - R_vector*R_dot_v
        u_parallel = R_vector*R_dot_v # (B, L, 3, 3)
        #
        v_new = v_perpendicular*torch.cos(R_angle).unsqueeze(-2) + R_cross_v*torch.sin(R_angle).unsqueeze(-2) + u_parallel # (B, L, 3, 3)
        #
        xyz = v_new + (xyz[:,:,1]+T).unsqueeze(-2)
        return xyz, state

class IterBlock(nn.Module):
    def __init__(self, d_msa=256, d_pair=128,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, p_drop=0.15,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        self.msa2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                                      n_head=n_head_msa,
                                      d_state=SE3_param['l0_out_features'],
                                      use_global_attn=use_global_attn,
                                      d_hidden=d_hidden_msa, p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                 d_hidden=d_hidden, p_drop=p_drop)
        self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair,
                                      d_hidden=d_hidden, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair,
                               d_state=SE3_param['l0_out_features'],
                               SE3_param=SE3_param,
                               p_drop=p_drop)

    def forward(self, msa, pair, xyz, state, idx, use_checkpoint=False):
        ca_xyz = xyz[:,:,1,:].contiguous()
        rbf_feat = rbf(torch.cdist(ca_xyz, ca_xyz))
        if use_checkpoint:
            msa = checkpoint.checkpoint(create_custom_forward(self.msa2msa), msa, pair, rbf_feat, state)
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair)
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair, rbf_feat)
            xyz, state = checkpoint.checkpoint(create_custom_forward(self.str2str, top_k=128), msa, pair, xyz, state, idx)
        else:
            msa = self.msa2msa(msa, pair, rbf_feat, state)
            pair = self.msa2pair(msa, pair)
            pair = self.pair2pair(pair, rbf_feat)
            xyz, state = self.str2str(msa.float(), pair.float(), xyz.float(), state.float(), idx, top_k=128) 
        
        return msa, pair, xyz, state

class IterativeSimulator(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4,
                 d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 p_drop=0.15):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block
        
        # Initial structure update
        self.proj_state = nn.Linear(22, SE3_param['l0_out_features'])
        self.init_str2str = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                    d_state=SE3_param['l0_out_features'],
                                    SE3_param=SE3_param,
                                    p_drop=p_drop)

        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.ModuleList([IterBlock(d_msa=d_msa_full, d_pair=d_pair,
                                                        n_head_msa=n_head_msa,
                                                        n_head_pair=n_head_pair,
                                                        d_hidden_msa=8,
                                                        d_hidden=d_hidden,
                                                        p_drop=p_drop,
                                                        use_global_attn=True,
                                                        SE3_param=SE3_param)
                                                        for i in range(n_extra_block)])

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.ModuleList([IterBlock(d_msa=d_msa, d_pair=d_pair,
                                                       n_head_msa=n_head_msa,
                                                       n_head_pair=n_head_pair,
                                                       d_hidden=d_hidden,
                                                       p_drop=p_drop,
                                                       use_global_attn=False,
                                                       SE3_param=SE3_param)
                                                       for i in range(n_main_block)])

        # Final SE(3) refinement
        if n_ref_block > 0:
            self.str_refiner = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                       d_state=SE3_param['l0_out_features'],
                                       SE3_param=SE3_param,
                                       p_drop=p_drop)
    
        self.norm_state = nn.LayerNorm(SE3_param['l0_out_features'])
        self.pred_lddt = nn.Linear(SE3_param['l0_out_features'], 1)
        
        self.reset_parameter()

    def reset_parameter(self):
        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_state.bias)
        self.pred_lddt = init_lecun_normal(self.pred_lddt)
        nn.init.zeros_(self.pred_lddt.bias)
        
    def forward(self, msa, msa_full, pair, xyz, state, idx, use_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        
        state = self.proj_state(state)
        if use_checkpoint:
            xyz, state = checkpoint.checkpoint(create_custom_forward(self.init_str2str, top_k=128),
                                               msa, pair, xyz, state, idx)
        else:
            xyz, state = self.init_str2str(msa.float(), pair.float(), xyz.float(), state.float(), idx, top_k=128)

        for i_m in range(self.n_extra_block):
            msa_full, pair, xyz, state = self.extra_block[i_m](msa_full, pair,
                                                               xyz, state, idx,
                                                               use_checkpoint=use_checkpoint)

        for i_m in range(self.n_main_block):
            msa, pair, xyz, state = self.main_block[i_m](msa, pair,
                                                         xyz, state, idx,
                                                         use_checkpoint=use_checkpoint)
        
        for i_m in range(self.n_ref_block):
            xyz, state = self.str_refiner(msa.float(), pair.float(), xyz.float(), state.float(), idx, top_k=128)

        lddt = self.pred_lddt(self.norm_state(state)).squeeze(-1)

        return msa, pair, xyz, lddt
