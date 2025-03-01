import argparse
import os

from yacs.config import CfgNode as CN

cfg = CN()

cfg.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
cfg.dir_name = 'OFER'
cfg.device = 'cuda'
cfg.seed = 42
cfg.withseed = False
cfg.pretrained_model_path = os.path.join(cfg.root_dir, cfg.dir_name, 'pretrained', 'best.pt')
cfg.output_dir = os.path.join(cfg.root_dir, cfg.dir_name, 'output')
cfg.input_dir = os.path.join(cfg.root_dir, cfg.dir_name, 'input')
cfg.checkpoint_dir = os.path.join(cfg.root_dir, cfg.dir_name, 'checkpoints')
cfg.pretrainwithgrad = False
cfg.modelname = ''

cfg.model = CN()
cfg.model.testing = False
cfg.model.validation = False
cfg.model.name = ''
cfg.model.pretrainname = ''

cfg.model.flame23_model_path = os.path.join(cfg.root_dir, 'data/FLAME2023', 'flame2023_no_jaw.pkl')
cfg.model.static_landmark_embedding_path = os.path.join(cfg.root_dir, 'data/FLAME2023', 'flame_static_embedding.pkl')
cfg.model.dynamic_landmark_embedding_path = os.path.join(cfg.root_dir, 'data/FLAME2023', 'flame_dynamic_embedding.npy')
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.root_dir, 'data/FLAME2023', 'landmark_embedding.npy')
cfg.model.pretrained = './pretrained'
cfg.model.frontonly = False
cfg.model.usenewfaceindex = False
cfg.model.rankprune = 0
cfg.model.flametype = 'flame23'
cfg.model.nettype = 'default'
cfg.model.n_shape = 300
cfg.model.n_exp = 100
cfg.model.n_pose = 15
cfg.model.n_tex = 200
cfg.model.batch_size = 32
cfg.model.n_cam = 3
cfg.model.n_light = 27
cfg.model.uv_size = 256
cfg.model.tex_type = 'FLAME'
cfg.model.layers = 8
cfg.model.hidden_layers_size = 256
cfg.model.mapping_layers = 3
cfg.model.use_pretrained = True
cfg.model.landmark = False
cfg.model.uniform = False
cfg.model.use_reg = False
cfg.model.with_exp = False
cfg.model.with_val = False
cfg.model.with_unpose = False
cfg.model.classfree = False
cfg.model.expencoder = 'arcface'
cfg.model.preexpencoder = 'arcface'
cfg.model.prenettype = 'default'
cfg.model.with_lmk = False
cfg.model.istrial = False
cfg.model.sampling = 'ddpm'
cfg.model.numsamples = 100
cfg.model.with_freeze = 'l4'
cfg.model.arcface_pretrained_model = os.path.join(cfg.root_dir, 'data/pretrained/arcface/glint360_r100/backbone.pth')

cfg.net = CN()
cfg.net.shape_dim = 5023*3
cfg.net.flame_dim = 400
cfg.net.full_shape_dim = 5023*3
cfg.net.context_dim = 512
cfg.net.time_dim = 512
cfg.net.dims = []
cfg.net.predims = []
cfg.net.numqkv = 1
cfg.net.numattn = 1
cfg.net.numsamples = 100
cfg.net.arch = ''
cfg.net.rankarch = ''
cfg.net.residual = 'unet'
cfg.net.tag = 'ofer'
cfg.net.losstype = 'mse'
cfg.net.mode = 'sep'
cfg.net.with100 = False
cfg.net.with20 = False
cfg.net.numpoints = 5023

cfg.varsched = CN()
cfg.varsched.num_steps = 200
cfg.varsched.beta_1 = 1e-4
cfg.varsched.beta_T = 0.01
cfg.varsched.mode = 'linear'

cfg.dataset = CN()
#cfg.dataset.training_data = ['Stirling', 'FaceWarehouse']
cfg.dataset.training_data = ['Lyhm', 'Stirling', 'FaceWarehouse', 'Coma']
cfg.dataset.validation_data = ['Florence']
cfg.dataset.validation_exp_data = ['AFLW2000']
cfg.dataset.test_data = ['NOW']
cfg.dataset.lmk = 'insight'
cfg.dataset.batch_size = 32
cfg.dataset.n_images = 4
cfg.dataset.tocenter = False
cfg.dataset.pretrained = './pretrained'
cfg.dataset.flametype = 'flame23'
cfg.dataset.flipchannels = False
cfg.dataset.with100 = False
cfg.dataset.with20 = False
cfg.dataset.with_unpose = False
cfg.dataset.occlusion = 0
cfg.dataset.arc224 = 0
cfg.dataset.resnethalfimg = 0
cfg.dataset.resnetfullimg = 0
cfg.dataset.epoch = 100000
cfg.dataset.num_workers = 4
cfg.dataset.root = os.path.join(cfg.root_dir, 'data')
cfg.dataset.topology_path = os.path.join(cfg.root_dir, 'data/FLAME2020', 'head_template.obj')
cfg.dataset.flame_template_vertices = os.path.join(cfg.root_dir, 'data/FLAME2020', 'head_template.npy')
cfg.dataset.flame_model_path = os.path.join(cfg.root_dir, 'data/FLAME2020', 'generic_model.pkl')
cfg.dataset.flame23_model_path = os.path.join(cfg.root_dir, 'data/FLAME2023', 'flame2023.pkl')
cfg.dataset.flame_lmk_embedding_path = os.path.join(cfg.root_dir, 'data/FLAME2023', 'landmark_embedding.npy')
cfg.dataset.n_shape = 300
cfg.dataset.n_exp = 100

#----------------------------
# Mask Weights
#----------------------------

cfg.mask_weights = CN()
cfg.mask_weights.face = 150.0
cfg.mask_weights.nose = 50.0
cfg.mask_weights.lips = 50.0
cfg.mask_weights.forehead = 50.0
cfg.mask_weights.lr_eye_region = 50.0
cfg.mask_weights.eye_region = 50.0
cfg.mask_weights.ears = 0.01
cfg.mask_weights.eyes = 0.01

cfg.mask_weights.whole = 1.0
cfg.running_average = 7

#---------------------------
# Training options
#-----------------------------

cfg.train = CN()
cfg.train.use_mask = True
cfg.train.max_epochs = 100000
cfg.train.lr = 1e-4
cfg.train.diff_lr = 1e-4
cfg.train.farl_lr = 1e-4
cfg.train.clip_lr = 1e-4
cfg.train.dinov2_lr = 1e-4
cfg.train.rank_lr = 1e-4
cfg.train.joint_lr = 1e-4
cfg.train.flame_lr = 1e-4
cfg.train.point_lr = 1e-4
cfg.train.varsched_lr = 1e-4
cfg.train.net_lr = 1e-4
cfg.train.fnet_lr = 1e-4
cfg.train.addnet_lr = 1e-4
cfg.train.arcface_lr = 1e-4
cfg.train.hse_lr = 1e-4
cfg.train.resnet_lr = 1e-4
cfg.train.rank_lr = 1e-4
cfg.train.weight_decay = 0.0
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.max_steps = 500000
cfg.train.val_steps = 10
cfg.train.checkpoint_steps = 1000
cfg.train.checkpoint_epochs_steps = 10000
cfg.train.val_save_img = 1200
cfg.train.vis_steps = 1200
cfg.train.val_vis_dir = 'val_images'
cfg.train.reset_optimizer = False
cfg.train.resume = False
cfg.train.resume_checkpoint = ''
cfg.train.resumepretrain = False
cfg.train.resume_pretrain_checkpoint = ''
cfg.train.write_summary = True


# TEST
cfg.test = CN()
cfg.test.num_points = 5023
cfg.test.batch_size = 1
cfg.test.point_dim = 3
cfg.test.cache_path = os.path.join(cfg.root_dir, 'data/NOW/cache') 
#cfg.test.now_images = os.path.join(cfg.root_dir, 'data/NOW/final_release_version/iphone_pictures') 
cfg.test.now_images = os.path.join(cfg.root_dir, 'data/NOW/arcface_input') 
cfg.test.stirling_hq_images = os.path.join(cfg.root_dir, 'data/STIRLING/arcface_input/HQ') 
cfg.test.stirling_lq_images = os.path.join(cfg.root_dir, 'data/STIRLING/arcface_input/LQ') 
cfg.test.affectnet_images = os.path.join(cfg.root_dir, 'data/AFFECTNET/arcface_input') 
cfg.test.aflw2000_images = os.path.join(cfg.root_dir, 'data/AFLW2000/arcface_input') 

def get_cfg_defaults():
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path', required=True)
    parser.add_argument('--checkpoint1', type=str, help='checkpoint1 location to load', default='shape')
    parser.add_argument('--checkpoint2', type=str, help='checkpoint2 location to load', default='experssion, jaw')
    parser.add_argument('--checkpoint3', type=str, help='checkpoint3 location to load', default='pose')
    parser.add_argument('--checkpoint4', type=str, help='checkpoint4 location to load', default='cam')
    parser.add_argument('--numcheckpoint', type=int, help='number of checkpoints', default=1)
    parser.add_argument('--test_dataset', type=str, help='Test dataset path', default='')
    parser.add_argument('--filename', type=str, help='filename', default='')
    parser.add_argument('--imagepath', type=str, help='imagepath', default='')
    parser.add_argument('--toseed', type=int, help='to seed or not', default=0)
    
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg, args
