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
    def __init__(self):
        super(Distiller, self).__init__()

        # ***** Embedding Function*****#

        # self.embed_query = Embed(768, 768)
        # self.embed_key = Embed(768, 768)
        # self.embed_value = Embed(768, 768)

        # # For different architecture
        # # self.embed_key = nn.Sequential(nn.Conv2d(opt.t_dim,opt.t_dim,2,2,0,bias=False),nn.BatchNorm2d(opt.t_dim))
        # self.embed_value = Embed(768, 768)

        self.head = 8

    def forward(self, f_s, f_t):

        q = f_s
        k = f_t
        v = f_s

        b, c, h, w = v.shape
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)  # (b,heads,hw,d)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.heads)  # (b,heads,hw,d)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.heads)  # (b,heads,hw,d)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # (b,heads,hw,hw)
        sim = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', sim, v)  # (b,heads,hw,d)

        out = rearrange(out, 'b h (x y) d ->b (h d) x y', x=h, y=w)  # (b,c,h,w)
        q = rearrange(q, 'b h (x y) d ->b (h d) x y', x=h, y=w)  # (b,c,h,w)
        k = rearrange(k, 'b h (x y) d ->b (h d) x y', x=h, y=w)  # (b,c,h,w)

        # loss = nn.MSELoss()(out,k) # student as out, teacher key
        loss = nn.MSELoss()(out, f_t)  # student as out, teacher feature
        return loss

