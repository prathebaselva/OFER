# inspired by MICA code

import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from jobs import trainExpGen

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

if __name__ == '__main__':
    from src.configs.config import parse_args

    cfg, args = parse_args()

    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.output_dir = os.path.join('./output', exp_name)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.empty_cache()
    num_gpus = torch.cuda.device_count()
    if args.toseed == 1:
        cfg.withseed = True

    mp.spawn(trainExpGen, args=(num_gpus, cfg), nprocs=num_gpus, join=True)

    exit(0)
