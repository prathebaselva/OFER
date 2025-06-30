# inspired by MICA code

import os
import sys
import torch
import re
import torch.backends.cudnn as cudnn
import numpy as np
import random
from pytorch_lightning import seed_everything

from src.test import Tester
from src.testerrank import Tester as TesterRank

from src.models.baselinemodels.flameparamdiffusion_model import FlameParamDiffusionModel
from src.models.baselinemodels.flameparamrank_model import FlameParamRankModel

def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        cudnn.deterministic = True
        cudnn.benchmark = False

if __name__ == '__main__':
    from src.configs.config import get_cfg_defaults, update_cfg
    from src.configs.config import parse_args
    deviceid = torch.cuda.current_device()
    torch.cuda.empty_cache()
    num_gpus = torch.cuda.device_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("num gpus = ", num_gpus)
    print("device_id", deviceid, flush=True)
    print("device = ", device, flush=True)

    cfg_file = './src/configs/config_flameparamrank_flame23.yml' 
    cfg_rank = get_cfg_defaults()
    if cfg_file is not None:
        cfg_rank = update_cfg(cfg_rank, cfg_file)



    cfg, args = parse_args()


    cfg_rank.train.resume_checkpoint = args.checkpoint1 
    cfg_rank.model.sampling = 'ddpm'
    cfg_rank.net.arch = 'archv4'
    cfg_rank.varsched.num_steps = 1000
    cfg_rank.varsched.beta_1 = 1e-4
    cfg_rank.varsched.beta_T = 1e-2
    cfg_rank.train.resume=True
    cfg_rank.train.resumepretrain = False
    cfg_rank.model.expencoder = 'arcfarl'
    cfg_rank.model.preexpencoder = 'arcface'
    cfg_rank.model.prenettype = 'preattn'
    cfg_rank.model.numsamples = 100
    cfg_rank.model.usenewfaceindex = True
    cfg_rank.model.istrial = False
    cfg_rank.net.losstype = 'Softmaxlistnetloss'
    cfg_rank.net.numattn = 1
    cfg_rank.net.predims = [300,50,10]
    cfg_rank.model.flametype = 'flame20'
    cfg_rank.dataset.flametype = 'flame20'
    cfg_rank.model.nettype = 'listnet'
    cfg_rank.net.rankarch = 'scorecb1listnet'
    cfg_rank.net.shape_dim = 5355
    cfg_rank.net.context_dim = 1024
    cfg_rank.model.testing = True
    seed_everything(1)
    model_rank = FlameParamRankModel(cfg_rank, 'cuda')

    testerrank = TesterRank(model_rank, cfg_rank, deviceid)
    testerrank.model.load_model()


    cfg1 = cfg.clone()
    cfg2 = cfg.clone()

    cfg1.model.sampling = 'ddim'
    cfg1.model.with_exp = False
    cfg1.model.expencoder = 'arcface'
    cfg1.net.flame_dim = 300
    cfg1.net.arch = 'archv4'
    cfg1.net.context_dim = 512
    cfg1.model.nettype = 'preattn'
    cfg1.net.dims = [300,50,10]
    cfg1.net.numattn = 1
    cfg1.train.resume = True
    cfg1.dataset.flametype = 'flame20'
    cfg1.model.flametype = 'flame20'
    cfg1.train.resume_checkpoint = args.checkpoint2
    cfg1.model.testing = True
    model1 = FlameParamDiffusionModel(cfg1, 'cuda')
    model1.eval()

    cfg2.model.sampling = 'ddpm'
    cfg2.model.with_exp = True
    cfg2.dataset.flametype = 'flame20'
    cfg2.model.flametype = 'flame20'
    cfg2.net.arch = 'archexp53'
    cfg2.net.context_dim = 1024
    cfg2.model.expencoder = 'arcfarl'
    cfg2.model.nettype = 'preattn'
    cfg2.net.flame_dim = 53
    cfg2.net.context_dim = 1024
    cfg2.model.n_exp = 50
    cfg2.net.dims = [53,25,10]
    cfg2.train.resume = True
    cfg2.train.resume_checkpoint = args.checkpoint3
    cfg2.model.testing = True
    model2 = FlameParamDiffusionModel(cfg2, 'cuda')
    model2.eval()


    models = [model1, model2]
    cfgs = [cfg1, cfg2]
    cfg.dataset.flametype = 'flame20'
    cfg.model.flametype = 'flame20'

    cfg.output_dir = args.outputpath 
    cfg.input_dir = args.imagepath
    tester = Tester(models, cfg, cfgs, deviceid, args, testerrank)
    tester.test_realocc(name='realocc', id='nocache', numface=100, numexp=15)

