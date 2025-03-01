import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from .layers import ScaleShiftGNMLPLayerRank 

import math


class RankMLPNet(Module):
    def __init__(self, config, device):
        super().__init__()
        self.shape_dim = config.shape_dim
        self.context_dim = config.context_dim
        self.device = device

        self.relu = torch.nn.ReLU()
        self.ranklayers = ModuleList([
            ScaleShiftGNMLPLayerRank(2*self.shape_dim, 4096, self.context_dim, 8),
            ScaleShiftGNMLPLayerRank(4096, 1024, self.context_dim, 8),
            ScaleShiftGNMLPLayerRank(1024, 256, self.context_dim, 8),
            ScaleShiftGNMLPLayerRank(256, 64, self.context_dim, 8),
            ScaleShiftGNMLPLayerRank(64, 1, self.context_dim, islast=True)])
        self.rank_skip_layers = []
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, context, numsamples):
        batch_size = x.size(0)
        splitsize = int(batch_size / numsamples)

        unet_out = []
        out = x.clone()
        k = 1

        out = x.clone()
        out = out.reshape(batch_size*numsamples, -1).to(self.device)
        context = torch.tile(context, (1,1, numsamples)).reshape(context.shape[0]*numsamples, -1)
        for layer in self.ranklayers:
            out = layer(ctx=context, x=out)
        out = out.view(batch_size, -1)
        sout = self.softmax(out)
        return sout, out

