import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math
from einops import rearrange
from torch import einsum
from torch.nn.modules import loss
import logging

'''Set up a module logger'''
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(dim_out)  # Normalize(2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x)
        return x


class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()
        self.beta = args.beta
        self.gamma = args.gamma
        # assert (self.beta >0.0 or self.gamma >0.0), \
        #    'If you really want to train model without distillation,\
        #    comment this assertation or use other file.'

        # if self.beta == 0.0:
        #     logger.info("***Important Note: Seq Distillation Has Been Removed!")
        # if self.gamma == 0.0:
        #     logger.info("***Important Note: Anchor Distillation Has Been Removed!")

        self.heads = args.heads
        self.n = args.n
        self.m = args.m
        self.sp_h = args.sp_h
        self.sp_w = args.sp_w
        self.sp_hs = args.sp_h if args.sp_hs == 0 else args.sp_hs
        self.sp_ws = args.sp_w if args.sp_ws == 0 else args.sp_ws
        self.anchor_h = args.anchor_h
        self.anchor_w = args.anchor_w
        self.attn_type = args.attn_type
        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        # ***** Embedding Function*****#
        self.embed_query = Embed(s_channels[-1], t_channels[-1])
        self.embed_key = Embed(t_channels[-1], t_channels[-1])
        # For different architecture
        # self.embed_key = nn.Sequential(nn.Conv2d(opt.t_dim,opt.t_dim,2,2,0,bias=False),nn.BatchNorm2d(opt.t_dim))
        self.embed_value = Embed(s_channels[-1], t_channels[-1])

        self.t_net = t_net
        self.s_net = s_net

        self.loss_divider = [8, 4, 2, 1, 1, 4 * 4]

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        feat_num = len(t_feats)
        s_feats, s_out = self.s_net.extract_feature(x)

        loss_distill = 0
        loss_distill = [self.distillation_loss_all(f_s, f_t.detach()) for f_s, f_t in zip(s_feats[-1], t_feats[-1])]

        if type(loss_distill) == list:
            loss_distill = sum([i for i in loss_distill])
        return s_out, loss_distill

    def sub_loss(self, q, k, v, f_t, heads):
        # multi-head, heads*c = d
        if self.attn_type == 'spatial_attn':
            heads = ((self.n - self.sp_h) // self.sp_hs + 1) * \
                    ((self.m - self.sp_w) // self.sp_ws + 1)
        # else:
        #     heads = self.heads

        b, c, h, w = v.shape
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=heads)  # (b,heads,hw,d)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=heads)  # (b,heads,hw,d)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)  # (b,heads,hw,d)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # (b,heads,hw,hw)
        sim = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', sim, v)  # (b,heads,hw,d)

        out = rearrange(out, 'b h (x y) d ->b (h d) x y', x=h, y=w)  # (b,c,h,w)
        q = rearrange(q, 'b h (x y) d ->b (h d) x y', x=h, y=w)  # (b,c,h,w)
        k = rearrange(k, 'b h (x y) d ->b (h d) x y', x=h, y=w)  # (b,c,h,w)

        # loss = nn.MSELoss()(out,k) # student as out, teacher key
        loss = nn.MSELoss()(out, f_t)  # student as out, teacher feature
        return loss

    def distillation_patchgroup(self, q, k, v, f_t):
        ''' f_t: B,N,H,W
            n: n patches along h axis
            m: m patches along w axis
        '''

        ''' Slice tensor'''
        if self.attn_type == 'spatial_attn':
            q = self.grid_tensor_spatial(q, self.n, self.m, self.sp_h, self.sp_w, self.sp_hs, self.sp_ws)
            k = self.grid_tensor_spatial(k, self.n, self.m, self.sp_h, self.sp_w, self.sp_hs, self.sp_ws)
            v = self.grid_tensor_spatial(v, self.n, self.m, self.sp_h, self.sp_w, self.sp_hs, self.sp_ws)
            f_t = self.grid_tensor_spatial(f_t, self.n, self.m, self.sp_h, self.sp_w, self.sp_hs, self.sp_ws)
        else:
            q = self.grid_tensor_seq(q, self.n, self.m)
            k = self.grid_tensor_seq(k, self.n, self.m)
            v = self.grid_tensor_seq(v, self.n, self.m)
            f_t = self.grid_tensor_seq(f_t, self.n, self.m)

        if self.attn_type == 'stack_attn' or self.attn_type == 'spatial_attn':
            ''' Patch-group distillation'''
            q = q.permute(0, 2, 1, 3, 4).reshape(q.size(0), -1, q.size(3), q.size(4))
            k = k.permute(0, 2, 1, 3, 4).reshape(k.size(0), -1, k.size(3), k.size(4))
            v = v.permute(0, 2, 1, 3, 4).reshape(v.size(0), -1, v.size(3), v.size(4))
            f_t = f_t.permute(0, 2, 1, 3, 4).reshape(f_t.size(0), -1, f_t.size(3), f_t.size(4))
            total_loss = self.sub_loss(q, k, v, f_t, self.heads)
        elif self.attn_type == 'batch_attn':
            q = q.permute(0, 2, 1, 3, 4).reshape(-1, q.size(1), q.size(3), q.size(4))
            k = k.permute(0, 2, 1, 3, 4).reshape(-1, k.size(1), k.size(3), k.size(4))
            v = v.permute(0, 2, 1, 3, 4).reshape(-1, v.size(1), v.size(3), v.size(4))
            f_t = f_t.permute(0, 2, 1, 3, 4).reshape(-1, f_t.size(1), f_t.size(3), f_t.size(4))
            # print(q.shape)
            total_loss = self.sub_loss(q, k, v, f_t, self.heads)
        elif self.attn_type == None or self.attn_type == 'None':
            # forward patch-level feat n*m times
            total_loss = 0
            for i in range(q.size(2)):
                total_loss += self.sub_loss(q[:, :, i], k[:, :, i], v[:, :, i], f_t[:, :, i], self.heads)

        return total_loss

    def distillation_anchorpoint(self, q, k, v, f_t):
        # Anchor point
        kernel_size = [q.size(2) // self.anchor_h, q.size(3) // self.anchor_w]
        stride_size = kernel_size
        q = torch.nn.AvgPool2d(kernel_size, stride_size)(q)
        k = torch.nn.AvgPool2d(kernel_size, stride_size)(k)
        v = torch.nn.AvgPool2d(kernel_size, stride_size)(v)
        f_t = torch.nn.AvgPool2d(kernel_size, stride_size)(f_t)

        anchor_loss = self.sub_loss(q, k, v, f_t, heads=1)
        return anchor_loss

    def distillation_loss_all(self, f_s, f_t):

        if len(f_s.size()) == 3:
            f_s = f_s.unsqueeze(0)
        if len(f_t.size()) == 3:
            f_t = f_t.unsqueeze(0)
        q = self.embed_query(f_s)  # (b,c,h,w)

        k = self.embed_key(f_t)  # (b,c,h,w)
        # k = f_t # no 3x3

        v = self.embed_value(f_s)  # (b,c,h,w), student

        nxm_loss = self.distillation_patchgroup(q, k, v, f_t)
        anchor_loss = self.distillation_anchorpoint(q, k, v, f_t)
        return self.beta * nxm_loss + self.gamma * anchor_loss

    def get_embed_params(self):
        m = []
        m += self.embed_key.parameters()
        m += self.embed_query.parameters()
        m += self.embed_value.parameters()
        return m

    def grid_tensor_seq(self, x, n, m):
        '''
            Return a sequence of patches.
        '''
        b, c, h, w = x.shape
        k_size_h = h // n
        stride_h = k_size_h

        k_size_w = w // m
        stride_w = k_size_w

        temp = x.unfold(2, k_size_h, stride_h).unfold(3, k_size_w, stride_w)
        return temp.contiguous().view(temp.size(0), temp.size(1), -1, temp.size(4), temp.size(5))

    def grid_tensor_spatial(self, x, n, m, sp_h, sp_w, sp_hs, sp_ws):
        '''
            Not used by this work.
        '''
        b, c, h, w = x.shape
        kernel_h = h // n
        stride_h = kernel_h

        kernel_w = w // m
        stride_w = kernel_w

        temp = x.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w)

        temp = temp.permute(0, 1, 4, 5, 2, 3)
        temp = temp.unfold(4, sp_h, sp_hs).unfold(5, sp_w, sp_ws)
        temp = temp.permute(0, 1, 4, 5, 6, 7, 2, 3)
        temp = temp.contiguous().view(temp.size(0), temp.size(1), -1, temp.size(6), temp.size(7))
        return temp