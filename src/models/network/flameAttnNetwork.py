import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim_out)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim_out)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class PreCrossNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, ctx):
        x = self.norm(x)
        return self.fn(x, ctx)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        #self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Conv1d(dim, dim_out, 1)
        #self.proj = nn.Linear(dim, dim_out)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, ctx_emb_dim = None, groups = 8):
        super().__init__()
        all_emb_dim = time_emb_dim + ctx_emb_dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(all_emb_dim, dim_out * 2)
        )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, ctx_emb = None):
        scale_shift = None
        print(ctx_emb.shape, flush=True)
        all_emb = torch.cat([time_emb, ctx_emb], dim=1)
        if exists(self.mlp) and exists(all_emb):
            all_emb = self.mlp(all_emb)
            all_emb = rearrange(all_emb, 'b c -> b c 1')
            scale_shift = all_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) -> b h c', h = self.heads), qkv)

        q = q.softmax(dim = -1)
        k = k.sigmoid()

        q = q * self.scale        

        context = torch.einsum('b h d, b h e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d -> b h e', context, q)
        out = rearrange(out, 'b h c -> b (h c)', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, numqkv=16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.numqkv = numqkv

        self.to_qkv = []
        for i in range(numqkv):
            (self.to_qkv).append(nn.Conv1d(dim, hidden_dim * 3, 1, bias = False))
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.to_out_linear = nn.Linear(numqkv, 1)

    def forward(self, x):
        b, c, n = x.shape
        q = []
        k = []
        v = []

        for i in range(self.numqkv):
           self.qkv = self.to_qkv[i].to('cuda')
           qkv = self.qkv(x).chunk(3, dim = 1)
           q1, k1, v1 = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
           q.append(q1)
           k.append(k1)
           v.append(v1)
        q = torch.stack(q,dim=-2).squeeze(-1)
        k = torch.stack(k,dim=-2).squeeze(-1)
        v = torch.stack(v,dim=-2).squeeze(-1)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)
        out = self.to_out_linear(out)
        return out


class LinearCrossAttention(nn.Module):
    def __init__(self, dim, ctx_dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(ctx_dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(ctx_dim, hidden_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            RMSNorm(dim)
        )

    def forward(self, x, ctx):
        b, c = x.shape
        q = self.to_q(x)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        q = rearrange(q, 'b (h c) -> b h c', h = self.heads)
        k = rearrange(k, 'b (h c) -> b h c', h = self.heads)
        v = rearrange(v, 'b (h c) -> b h c', h = self.heads)

        q = q.softmax(dim = -1)
        k = k.sigmoid()

        q = q * self.scale        

        context = torch.einsum('b h d, b h e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d -> b h e', context, q)
        out = rearrange(out, 'b h c-> b (h c)', h = self.heads)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, ctx_dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Linear(dim, hidden_dim , bias = False)
        self.to_k = nn.Linear(ctx_dim, hidden_dim , bias = False)
        self.to_v = nn.Linear(ctx_dim, hidden_dim , bias = False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, ctx):
        b, c = x.shape
        q = self.to_q(x)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        q = rearrange(q, 'b (h c) -> b h c', h = self.heads)
        k = rearrange(k, 'b (h c) -> b h c', h = self.heads)
        v = rearrange(v, 'b (h c) -> b h c', h = self.heads)

        q = q * self.scale

        sim = einsum('b h d, b h d -> b h', q, k)
        attn = sim.sigmoid()
        out = einsum('b h, b h d -> b h d', attn, v)

        out = rearrange(out, 'b h d -> b (h d)')
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        config,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        resnet_block_groups = 1,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        # determine dimensions
        dim = config.flame_dim
        out_dim = config.flame_dim 
        context_dim = config.context_dim

        self.channels = config.flame_dim
        input_channels = config.flame_dim

        init_dim = config.flame_dim
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1)
        self.numattn = config.numattn

        dims = config.dims 
        in_out = list(zip(dims[:-1], dims[1:]))
        numqkv = config.numqkv

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = 512

        sinu_pos_emb = SinusoidalPosEmb(time_dim, theta = sinusoidal_pos_emb_theta)
        fourier_dim = time_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.finals = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, ctx_emb_dim = context_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, ctx_emb_dim = context_dim),
                Residual(PreNorm(dim_in, Attention(dim_in, numqkv=numqkv))),
                Downsample(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, ctx_emb_dim = context_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, numqkv=numqkv)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, ctx_emb_dim = context_dim)

        for ind, (dim_out, dim_in) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1) 

            self.ups.append(nn.ModuleList([
                Upsample(dim_in, dim_out),
                block_klass(dim_out*2, dim_out, time_emb_dim = time_dim, ctx_emb_dim = context_dim),
                block_klass(dim_out*2, dim_out, time_emb_dim = time_dim, ctx_emb_dim = context_dim),
                Residual(PreNorm(dim_out, Attention(dim_out, numqkv=numqkv)))
            ]))


        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.finals.append(nn.ModuleList([
            block_klass(dim * 2, dim, time_emb_dim = time_dim, ctx_emb_dim = context_dim),
            Residual(PreNorm(dim_out, Attention(dim_out, numqkv=numqkv))),
            nn.Conv1d(dim, self.out_dim, 1)
        ]))

    def forward(self, x, t, context):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = x.unsqueeze(-1)

        inittime = t

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(inittime)
        h = []

        if self.numattn == 1:
            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t, context)
                h.append(x)
                x = block2(x, t, context)
                x = attn(x)
                h.append(x)
                x = x.squeeze()
                x = downsample(x)
                x = x.unsqueeze(-1)
            x = self.mid_block1(x, t, context)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t, context)
            for upsample, block1, block2, attn in self.ups:
                x = x.squeeze()
                x = upsample(x)
                x = x.unsqueeze(-1)
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, t, context)
                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t, context)
                x = attn(x)
        else:
            for block1, block2, attn1, attn2, attn3, downsample in self.downs:
                x = block1(x, t, context)
                h.append(x)
                x = block2(x, t, context)
                x = attn1(x)
                x = attn2(x)
                x = attn3(x)
                h.append(x)
                x = x.squeeze()
                x = downsample(x)
                x = x.unsqueeze(-1)
            x = self.mid_block1(x, t, context)
            x = self.mid_attn1(x)
            x = self.mid_attn2(x)
            x = self.mid_attn3(x)
            x = self.mid_block2(x, t, context)
            for upsample, block1, block2, attn1, attn2, attn3 in self.ups:
                x = x.squeeze()
                x = upsample(x)
                x = x.unsqueeze(-1)
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, t, context)
                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t, context)
                x = attn1(x)
                x = attn2(x)
                x = attn3(x)

        x = torch.cat((x, r), dim = 1)

        for block, attn, linear in self.finals:
            x = block(x, t, context)
            x = attn(x)
            x = linear(x)
        return x.squeeze()

# gaussian diffusion trainer class
