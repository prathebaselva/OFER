# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2025 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ofer@tue.mpg.de


import os
import sys
from glob import glob

import cv2
from PIL import Image as PILImage
import numpy as np
import torch
import re
import torch.nn.functional as F
import torch.distributed as dist
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

from src.configs.config import cfg
from src.utils import util
from src.models.flame import FLAME
import trimesh
import scipy.io


sys.path.append("./src")
input_mean = 127.5
input_std = 127.5



class Tester(object):
    def __init__(self, models, config=None, cfgs=None, device=None, args=None, rankmodel=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.cfgs = cfgs

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
        self.args = args
        self.rankmodel = rankmodel

        import clip
        self.pretrainedfarlmodel = self.cfg.model.pretrained
        self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
        farl_state = torch.load(os.path.join(self.pretrainedfarlmodel, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth"))
        self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
        self.farlmodel = self.farlmodel.to(self.device)

        self.model = {i:None for i in range(len(models))}
        for i in range(len(models)):
            self.model[i] = models[i].to(self.device)
            self.model[i].testing = True
            self.model[i].eval()

        flameModel = FLAME(self.cfg.model).to(self.device)
        self.faces = flameModel.faces_tensor.cpu()
        logger.info(f'[INFO]            {torch.cuda.get_device_name(device)}')

    def load_checkpoint(self, model, ckpt_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        checkpoint = torch.load(ckpt_path, map_location)

        if 'arcface' in checkpoint:
            print("arcface")
            model.arcface.load_state_dict(checkpoint['arcface'])
        if 'farl' in checkpoint:
            print("farl")
            model.farlmodel.load_state_dict(checkpoint['farl'])
            model.arcface.load_state_dict(checkpoint['arcface'])
        if 'hseencoder' in checkpoint:
            print("hseencoder")
            model.hseencoder.load_state_dict(checkpoint['hseencoder'])
        if 'resnet' in checkpoint:
            print("resnet")
            model.resnet.load_state_dict(checkpoint['resnet'])
        if 'net' in checkpoint:
            print("net")
            model.net.load_state_dict(checkpoint['net'])
        if 'fnet' in checkpoint:
            print("fnet")
            model.fnet.load_state_dict(checkpoint['fnet'])
        if 'var_sched' in checkpoint:
            print("var_sched")
            print(checkpoint['var_sched'])
            model.var_sched.load_state_dict(checkpoint['var_sched'], strict=False)
        if 'diffusion' in checkpoint:
            print("diffusion")
            model.diffusion.load_state_dict(checkpoint['diffusion'], strict=False)

        print("done", flush=True)
        logger.info(f"[TESTER] Resume from {ckpt_path}")
        return model

    def load_model_dict(self, model_dict):
        dist.barrier()

        self.model.canonicalModel.load_state_dict(model_dict['canonicalModel'])
        self.model.arcface.load_state_dict(model_dict['arcface'])

    def get_name(self, best_model, id):
        if '_' in best_model:
            name = id if id is not None else best_model.split('_')[-1][0:-4]
        else:
            name = id if id is not None else best_model.split('/')[-1][0:-4]
        return name


    def load_shape_cfg(self, ckpt, cfg):
        self.model[0].with_exp = False
        self.model[0].expencoder = cfg.model.expencoder
        self.model[0].net.flame_dim = cfg.net.flame_dim
        self.model[0].net.expencoder = cfg.model.expencoder

    def load_cfg(self, cfg, best_model):
        self.model.var_sched.num_steps = cfg.varsched.num_steps
        self.model.var_sched.beta_1 = cfg.varsched.beta_1
        self.model.var_sched.beta_T = cfg.varsched.beta_T
        self.model.net.flame_dim = cfg.net.flame_dim
        self.model.net.arch = cfg.net.arch
        self.model.expencoder = cfg.model.expencoder
        self.model.with_exp = cfg.model.with_exp
        self.model.sampling = cfg.model.sampling

        self.model = self.load_checkpoint(best_model)
        self.model.var_sched.num_steps = cfg.varsched.num_steps
        self.model.var_sched.beta_1 = cfg.varsched.beta_1
        self.model.var_sched.beta_T = cfg.varsched.beta_T
        self.model.net.flame_dim = cfg.net.flame_dim
        self.model.net.arch = cfg.net.arch
        self.model.expencoder = cfg.model.expencoder
        self.model.with_exp = cfg.model.with_exp
        self.model.sampling = cfg.model.sampling

    def load_cfgs(self, ckpts, cfgs):
        num = len(ckpts)

        for i in range(num):
            print(i, flush=True)
            self.model[i].with_exp = cfgs[i].model.with_exp
            self.model[i].sampling = cfgs[i].model.sampling
            self.model[i].with_lmk = cfgs[i].model.with_lmk
            self.model[i].expencoder = cfgs[i].model.expencoder
            self.model[i].net.flame_dim = cfgs[i].net.flame_dim
            self.model[i].net.arch = cfgs[i].net.arch
            self.model[i].net.context_dim = cfgs[i].net.context_dim
            self.model[i].var_sched.num_steps = cfgs[i].varsched.num_steps
            self.model[i].var_sched.beta_1 = cfgs[i].varsched.beta_1
            self.model[i].var_sched.beta_T = cfgs[i].varsched.beta_T

            self.model[i] = self.load_checkpoint(self.model[i], ckpts[i])

            self.model[i].with_exp = cfgs[i].model.with_exp
            self.model[i].sampling = cfgs[i].model.sampling
            self.model[i].with_lmk = cfgs[i].model.with_lmk
            self.model[i].expencoder = cfgs[i].model.expencoder

            if i != 2:
                self.model[i].net.flame_dim = cfgs[i].net.flame_dim
                self.model[i].net.arch = cfgs[i].net.arch
                self.model[i].net.context_dim = cfgs[i].net.context_dim
                self.model[i].var_sched.num_steps = cfgs[i].varsched.num_steps
                self.model[i].var_sched.beta_1 = cfgs[i].varsched.beta_1
                self.model[i].var_sched.beta_T = cfgs[i].varsched.beta_T



    def test_realocc(self, name='test', id='nocache', numface=100, numexp=15):
        self.realocc(name, id, numface, numexp)


    def realocc(self, best_id, id, numface=100, numexp=15, istest='val'):
        logger.info(f"[TESTER]  validation has begun!")
        for i in range(len(self.model.keys())):
            self.model[i].eval()
        valread = open(self.args.filename,'r')
        allimages = []
        image_names = []
        for line in valread:
            line = line.strip()
            image_names.append(line)
            allimages.append(os.path.join(self.args.imagepath, line))
        valread.close()
        
        from pytorch_lightning import seed_everything
        r = np.random.randint(1000)
        seed_everything(r)
        for i in range(len(allimages)):
            images = allimages[i]
            image_name = image_names[i]
            image_name_noext = images[:-4]
            npyfile = (re.sub('jpg','npy', images))
            if not os.path.exists(npyfile):
                continue
            imagefarl = self.farl_preprocess(PILImage.open(images))
            arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to('cuda')
            matfile = re.sub('jpg', 'mat', re.sub('arcface_input', 'FLAME_parameters', images))
            img450 = imread(re.sub('arcface_input/images', 'images', images))
            kpt = None 
            origimage = imread(images)
            normimage = origimage / 255.
            normtransimage = normimage.transpose(2, 0, 1)


            result = {'origimage': origimage,
                    'normimage': normimage,
                    'normtransimage': normtransimage,
                    'arcface': arcface,
                    'imagefarl': imagefarl,
                    'imgname': image_name,
                    'imgname_noext': image_name_noext,
                    'best_id': best_id,
                    'id': id,
                    'numface': numface,
                    'numexp': numexp,  
                    'actor': '',
                    'type':'',
                    'kpt': kpt,
                    'img450': img450,
                    'outfile': ''}
            self.decode(result)

    def decode(self, input):
        print("in decode", flush=True)
        origimage = input['origimage']
        normimage = input['normimage']
        normtransimage = input['normtransimage']
        if 'uncutimage' in input:
            uncutimage = input['uncutimage']
        arcface = input['arcface']
        imagefarl = input['imagefarl']
        image_name = input['imgname']
        image_name_noext = input['imgname_noext']
        best_id = input['best_id']
        id = input['id']
        numface= input['numface']
        numexp=input['numexp']
        outfile= input['outfile']
        actor=input['actor']
        type=input['type']
        kpt=input['kpt']
        istest='val'

        interpolate = 224
        origimage_copy = origimage.copy()
        arcface_rank = arcface.clone()
        with torch.no_grad():
            
            arcface1 = arcface.tile(numface,1,1,1)
            img_tensor1 = torch.Tensor(normtransimage).tile(numface,1,1,1).to('cuda')
            imgfarl_tensor1 = torch.Tensor(imagefarl).tile(numface,1,1,1).to('cuda')
            codedict1 = self.model[0].encode(img_tensor1, arcface1, imgfarl_tensor1)
            opdict1 = self.model[0].decode(codedict1, 0, withpose=False)
            pred_flameparam1 = opdict1['pred_flameparam']
            if 'pred_mesh' in opdict1:
                pred_shape_meshes = opdict1['pred_mesh']
                pred_shape_lmk = self.model[0].flame.compute_landmarks(pred_shape_meshes)
            shape = pred_flameparam1[:,:300]
            print("num shape = ", shape.shape, flush=True)
            print("num shape = ", pred_shape_meshes.shape, flush=True)
            ######### GET BEST RANK #######################
            maxindex, sortindex = self.rankmodel.getmaxsampleindex(arcface_rank, normtransimage, imagefarl, pred_shape_meshes)
            print("numface = ", numface)
            arcface = arcface.tile(numface,1,1,1)
            img_tensor = torch.Tensor(normtransimage).tile(numface,1,1,1).to('cuda')
            imgfarl_tensor = torch.Tensor(imagefarl).tile(numface,1,1,1).to('cuda')
            codedict2 = self.model[1].encode(img_tensor, arcface, imgfarl_tensor)

            opdict2 = []
            self.model[1].testing = True
            print(self.model[1].cfg.net.flame_dim, flush=True)
            print(numface, flush=True)
           
            from pytorch_lightning import seed_everything
            loops = int(100/numface)
            for i in range(loops):
                r = np.random.randint(1000)
                seed_everything(r)
                opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[maxindex].tile(numface,1).float()))

        os.makedirs(os.path.join(self.cfg.output_dir, f'{outfile}','flamesample'), exist_ok=True)
        flame_dst_folder = os.path.join(self.cfg.output_dir, f'{outfile}', 'flamesample', actor, type)
        os.makedirs(os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample'), exist_ok=True)
        shape_dst_folder = os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample', actor, type)
        os.makedirs(flame_dst_folder, exist_ok=True)
        os.makedirs(shape_dst_folder, exist_ok=True)
        print(flame_dst_folder, flush=True)

        image_name = re.sub('arcface_input/','',image_name)
        a = image_name
        savepath = os.path.split(os.path.join(flame_dst_folder, a))[0]
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.split(os.path.join(shape_dst_folder, a))[0]
        os.makedirs(savepath, exist_ok=True)

        pred_front = []
        all_pred_flame_meshes = []
        for ni in range(loops):
            if 'pred_mesh' in opdict2[ni]:
                pred_flame_meshes = opdict2[ni]['pred_mesh']
                ##### Get the shapes with maximum variations ############
                findices = np.load('pretrained/frontindices.npy') 
                pred_front.append(pred_flame_meshes[:,findices].reshape(numface,-1))
                all_pred_flame_meshes.append(pred_flame_meshes)
            else:
                continue
        pred_front = torch.vstack(pred_front)
        all_pred_flame_meshes = torch.vstack(all_pred_flame_meshes)

        expsortindex = np.arange(numexp)

        for num in range(numexp):
            currname = a[:-4]+'_'+str(0)+'_'+str(num)+'.jpg'
            savepath = os.path.join(flame_dst_folder, currname.replace('jpg', 'ply'))
            print(savepath, flush=True)
            trimesh.Trimesh(vertices=all_pred_flame_meshes[expsortindex[num]].cpu() * 1000.0, faces=self.faces, process=False).export(savepath)

        for num in range(1):
            currname = a[:-4]+'_'+str(num)+'.jpg'
            saveshapepath = os.path.join(shape_dst_folder, currname.replace('jpg', 'ply'))
            trimesh.Trimesh(vertices=pred_shape_meshes[maxindex[num]].cpu() * 1000.0, faces=self.faces, process=False).export(saveshapepath)

