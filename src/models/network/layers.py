import torch
from torch.nn import Module, Linear
import torch.nn as nn
import numpy as np



class ScaleShiftGNMLPLayerRank(Module):
    def __init__(self, dim_in, dim_out, dim_context, groups=8, islast=False):
        super(ScaleShiftGNMLPLayerRank, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.scale = None
        self.shift = None
        dim_ctx = dim_context 
        self.scale = Linear(dim_ctx, dim_out)
        self.shift = Linear(dim_ctx, dim_out, bias=False)
        self.silu = torch.nn.SELU()
        self.islast = islast
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)

    def forward(self, ctx, x):
        context = ctx.to("cuda")
        s = self.scale(context)
        scale = torch.sigmoid(s)
        shift = self.shift(context)
        if self.islast:
            return self.mlp(x) * scale + shift
        if x.shape[0] > 1:
            ret = self.gn(self.silu(self.mlp(x))) * scale + shift
        else:
            ret = (self.silu(self.mlp(x))) * scale + shift
        return ret

