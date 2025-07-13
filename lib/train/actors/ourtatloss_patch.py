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
    def __init__(self, args):
        super(Distiller, self).__init__()
        self.beta = 0.5
        self.gamma = 0.5

        self.heads = 8
        self.n = 8
        self.m = 8
        self.sp_h = 8
        self.sp_w = 8
        self.sp_hs = 8
        self.sp_ws = 8
        self.anchor_h = 8
        self.anchor_w = 8
        self.attn_type = 'spatial_attn'

        # ***** Embedding Function*****#
        # self.embed_query = Embed(s_channels[-1], t_channels[-1])
        # self.embed_key = Embed(t_channels[-1], t_channels[-1])
        # self.embed_value = Embed(s_channels[-1], t_channels[-1])

        # self.loss_divider = [8, 4, 2, 1, 1, 4 * 4]

    def forward(self, f_s, f_t):

        loss_distill = self.distillation_loss_all(f_s, f_t)

        return loss_distill

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

        # q = self.embed_query(f_s)  # (b,c,h,w)
        # k = self.embed_key(f_t)  # (b,c,h,w)
        # v = self.embed_value(f_s)  # (b,c,h,w), student
        q = f_s
        k = f_t
        v = f_s

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
        kernel_h = h // n   # 24 / 8
        stride_h = kernel_h  # 24 / 8

        kernel_w = w // m  # 24 / 8
        stride_w = kernel_w  # # 24 / 8

        temp = x.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w)

        temp = temp.permute(0, 1, 4, 5, 2, 3)
        temp = temp.unfold(4, sp_h, sp_hs).unfold(5, sp_w, sp_ws)
        temp = temp.permute(0, 1, 4, 5, 6, 7, 2, 3)
        temp = temp.contiguous().view(temp.size(0), temp.size(1), -1, temp.size(6), temp.size(7))
        return temp