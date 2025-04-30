# inspired by MICA code

import os
import sys
import torch
import re
import torch.backends.cudnn as cudnn
import numpy as np
import random
from pytorch_lightning import seed_everything

from src.testerrank import Tester as TesterRank
from src.tester import Tester2


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

    cfg_rank.train.resume_checkpoint = './output/config_flameparamrank_flame23/best_models/model_train_flameparamrank_arcfarl_lr1e6_flame23_listnetrank_crossattn_scorecb1_Softmaxloss_ddpm_samp100_final_all_resume_best.tar' 
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
    cfg_rank.model.flametype = 'flame23'
    cfg_rank.dataset.flametype = 'flame23'
    cfg_rank.model.nettype = 'listnet'
    cfg_rank.net.rankarch = 'scorecb1listnet'
    cfg_rank.net.shape_dim = 5355
    cfg_rank.net.context_dim = 1024
    cfg_rank.model.testing = True
    #r = np.random.randint(1000000)
    seed_everything(1)
    model_rank = FlameParamRankModel(cfg_rank, 'cuda')

    testerrank = TesterRank(model_rank, cfg_rank, deviceid)
    testerrank.model.load_model()
    #testerrank.model.eval()

    cfg, args = parse_args()
    if args.numcheckpoint > 1:
        from src.models.baselinemodels.meshdiffusion_resnet_cam_model import MeshDiffusionResnetCamModel
        cfg1 = cfg.clone()
        cfg2 = cfg.clone()
        cfg3 = cfg.clone()

        cfg1.model.sampling = 'ddim'
        cfg1.model.with_exp = False
        cfg1.model.expencoder = 'arcface'
        cfg1.net.flame_dim = 300
        cfg1.net.arch = 'archv4'
        cfg1.net.context_dim = 512
        cfg1.model.nettype = 'preattn'
        #cfg1.net.dims = [300,200,100,50]
        cfg1.net.dims = [300,50,10]
        cfg1.net.numattn = 1
        cfg1.train.resume = True
        cfg1.dataset.flametype = 'flame20'
        cfg1.model.flametype = 'flame20'
        cfg1.train.resume_checkpoint = args.checkpoint1
        #cfg1.model.flame20_model_path = '../FACEDATA/FLAME2020/generic_model.pkl'
        #cfg1.model.flame20_model_path = '../FACEDATA/FLAME2023/flame2023.pkl'
        cfg1.model.testing = True
        model1 = FlameParamDiffusionModel(cfg1, 'cuda')
        model1.eval()

        cfg2.model.sampling = 'ddpm'
        cfg2.model.with_exp = True
        #cfg2.model.flame20_model_path = '../FACEDATA/FLAME2023/flame2023.pkl'
        #cfg2.model.flame_model_path = '../FACEDATA/FLAME2020/generic_model.pkl'
        #cfg2.net.arch = 'archexp1'
        cfg2.dataset.flametype = 'flame20'
        cfg2.model.flametype = 'flame20'
        cfg2.net.arch = 'archexp53'
        cfg2.net.context_dim = 1024
        cfg2.model.expencoder = 'arcfarl'
        cfg2.model.nettype = 'preattn'
        cfg2.net.flame_dim = 53
        cfg2.net.context_dim = 1024
        cfg2.model.n_exp = 50
        #cfg2.dataset.n_exp = 50
        cfg2.net.dims = [53,25,10]
        cfg2.train.resume = True
        #cfg2.model.nettype = 'default'
        #cfg2.net.flame_dim = 103
        #cfg2.model.n_exp = 100
        #cfg2.dataset.n_exp = 100
        #cfg2.net.dims = [103,103,50,25]
        cfg2.train.resume_checkpoint = args.checkpoint2
        cfg2.model.testing = True
        model2 = FlameParamDiffusionModel(cfg2, 'cuda')
        model2.eval()

        cfg3.model.sampling = 'dd'
        cfg3.model.with_exp = True
        cfg3.net.arch = 'decoderv2'
        cfg3.net.flame_dim = 3
        cfg3.model.expencoder = 'arcface'
        cfg3.model.nettype = 'default'
        cfg3.train.resume = True
        cfg3.dataset.flametype = 'flame20'
        cfg3.model.flametype = 'flame20'
        cfg3.train.resume_checkpoint = args.checkpoint3
        print("checkpoint = ", os.path.exists(args.checkpoint3))
        
        models = [model1, model2]
        cfgs = [cfg1, cfg2]
        cfg.dataset.flametype = 'flame20'
        cfg.model.flametype = 'flame20'

        cfg.output_dir = 'pickpix_53_jaw05' 
        cfg.input_dir = args.imagepath
        tester = Tester2(models, cfg, cfgs, deviceid, args, testerrank)
        tester.test_aflw2000('realocc', 'nocache', 100)

    else:
        cfg.dataset.flametype = 'flame20'
        cfg.model.flametype = 'flame20'
        cfg.model.with_exp = False
        cfg.model.testing = True
        cfg.dataset.occlusion = 0
        cfg.dataset.flipchannels = False
        cfg.net.flame_dim = 300
        cfg.varsched.num_steps = 1000
        cfg.varsched.beta_1 = 1e-4
        cfg.varsched.beta_T = 1e-2
        cfg.train.resume=True
        cfg.train.resume_checkpoint = args.checkpoint1
        cfg.model.expencoder = 'arcface'
        cfg.net.context_dim = 512
        cfg.net.arch = 'preattn'
        cfg.model.sampling = 'ddpm'
        cfg.model.nettype = 'preattn'
        cfg.net.numattn = 1
        cfg.net.numqkv = 16
        cfg.net.dims=[300,50,10]
    
        model = FlameParamDiffusionModel(cfg, deviceid)
        tester = Tester(model, cfg, deviceid, args)
        #tester.test_now('flameparamdiffusion_arcface_flame20_attn_qkv16_samp100', 'nocache', 100, istest='val', isocc=0, filename=args.filename)
        cfg.output_dir = 'realocc'
        cfg.input_dir = 'arcface_input' 
        tester.test_realocc('diverse3d_flame20_attn_qkv16',  100)
