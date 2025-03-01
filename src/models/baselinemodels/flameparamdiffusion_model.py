# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#


import os
import sys


import torch
import torch.nn.functional as F

from src.models.arcface import Arcface
from src.models.base_model import BaseModel
from src.models.network.flameparamdiffusion_network import FlameParamDiffusion, VarianceScheduleMLP
from src.models.network.flameAttnNetwork import Unet 

from loguru import logger
import numpy as np
import trimesh

class FlameParamDiffusionModel(BaseModel):
    def __init__(self, config=None, device=None):
        super(FlameParamDiffusionModel, self).__init__(config, device, 'FlameParamDiffusionModel')
        self.expencoder = self.cfg.model.expencoder
        self.testing = self.cfg.model.testing
        self.validation = self.cfg.model.validation
        self.initialize()

    def create_model(self, model_config):
        mapping_layers = model_config.mapping_layers
        pretrained_path = None
        if not model_config.use_pretrained:
            pretrained_path = model_config.arcface_pretrained_model
        print("freeze = {}".format(self.cfg.model.with_freeze), flush=True)
        if self.expencoder == 'arcface':
            logger.info(f'[{self.tag}] creating arcface')
            print("device = ", self.device)
            self.arcface = Arcface(pretrained_path=pretrained_path, freeze=self.cfg.model.with_freeze).to(self.device)
        elif self.expencoder == 'clip':
            import clip
            logger.info(f'[{self.tag}] creating clip')
            self.clipmodel, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clipmodel = self.clipmodel.to(self.device)
            self.clipmodel = self.clipmodel.float()

        elif self.expencoder == 'dinov2':
            self.dinov2model = torch.hub.load('pretrained/dinov2', 'dinov2_vitl14', source='local', pretrained=False)
            self.dinov2model.load_state_dict(torch.load('pretrained/dinov2_vitl14.pth'))
            logger.info(f'[{self.tag}] creating dinov2')
            self.dinov2model = self.dinov2model.to(self.device)
            self.dinov2model = self.dinov2model.float()
        elif (self.expencoder == 'farl'):
            import clip
            self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
            farl_state = torch.load(os.path.join(model_config.pretrained, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth" ))
            self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
            self.farlmodel = self.farlmodel.to(self.device)
        elif self.expencoder == 'arcfarl':
            import clip
            self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
            farl_state = torch.load(os.path.join(model_config.pretrained, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth" ), map_location='cpu')
            self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
            self.farlmodel = self.farlmodel.to(self.device)
            self.arcface = Arcface(pretrained_path=pretrained_path, freeze=self.cfg.model.with_freeze).to(self.device)
        
        #### Diffusion Network
        self.net = Unet(config=self.cfg.net)
        self.var_sched = VarianceScheduleMLP(config=self.cfg.varsched)
        self.diffusion = FlameParamDiffusion(net=self.net,var_sched=self.var_sched, device=self.device, tag=self.cfg.net.tag, nettype=self.cfg.model.nettype)

    def load_for_test(self, model_path):
        if os.path.exists(model_path):
            logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path)
            if 'arcface' in checkpoint:
                self.arcface.load_state_dict(checkpoint['arcface'])
            if 'farl' in checkpoint:
                self.farlmodel.load_state_dict(checkpoint['farl'])
            if 'clip' in checkpoint:
                self.clipmodel.load_state_dict(checkpoint['clip'])
            if 'dinov2' in checkpoint:
                self.dinov2model.load_state_dict(checkpoint['dinov2'])
            if 'farlencoder' in checkpoint:
                self.farlencoder.load_state_dict(checkpoint['farlencoder'])
            if 'net' in checkpoint:
                self.net.load_state_dict(checkpoint['net'], strict=False)
            if 'var_sched' in checkpoint:
                self.var_sched.load_state_dict(checkpoint['var_sched'])
            if 'diffusion' in checkpoint:
                self.diffusion.load_state_dict(checkpoint['diffusion'], strict=False)
        else:
            logger.info(f'[{self.tag}] Checkpoint not available starting from scratch!')
            exit()

    def load_model(self):
        if self.cfg.train.resume:
            model_path = os.path.join(self.cfg.train.resume_checkpoint)
            if os.path.exists(model_path):
                logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
                checkpoint = torch.load(model_path)
                if 'arcface' in checkpoint:
                    print('arcface')
                    self.arcface.load_state_dict(checkpoint['arcface'])
                if 'farl' in checkpoint:
                    print('farl')
                    self.farlmodel.load_state_dict(checkpoint['farl'])
                if 'clip' in checkpoint:
                    print('clip')
                    self.clipmodel.load_state_dict(checkpoint['clip'])
                if 'dinov2' in checkpoint:
                    print('dinov2')
                    self.dinov2model.load_state_dict(checkpoint['dinov2'])
                if 'farlencoder' in checkpoint:
                    self.farlencoder.load_state_dict(checkpoint['farlencoder'])
                if 'net' in checkpoint:
                    print("net")
                    if self.testing:
                        self.net.load_state_dict(checkpoint['net'], strict=False)
                    else:
                        self.net.load_state_dict(checkpoint['net'], strict=False)
                if 'var_sched' in checkpoint:
                    self.var_sched.load_state_dict(checkpoint['var_sched'])
                if 'diffusion' in checkpoint:
                    print("diffusion")
                    if self.testing:
                        print("testing")
                        self.diffusion.load_state_dict(checkpoint['diffusion'], strict=False)
                    else:
                        self.diffusion.load_state_dict(checkpoint['diffusion'], strict=False)
            else:
                logger.info(f'[{self.tag}] Checkpoint {model_path} not available starting from scratch!')
                exit()

    def model_dict(self):
        if self.expencoder == 'arcface':
            return {
                'arcface': self.arcface.state_dict(),
                'net': self.net.state_dict(),
                'var_sched': self.var_sched.state_dict(),
                'diffusion': self.diffusion.state_dict(),
            }
        elif self.expencoder == 'arcfarl':
            return {
                'farl': self.farlmodel.state_dict(),
                'arcface': self.arcface.state_dict(),
                'net': self.net.state_dict(),
                'var_sched': self.var_sched.state_dict(),
                'diffusion': self.diffusion.state_dict(),
            }
        elif (self.expencoder == 'farl'):
            return {
                'farl': self.farlmodel.state_dict(),
                'net': self.net.state_dict(),
                'var_sched': self.var_sched.state_dict(),
                'diffusion': self.diffusion.state_dict(),
            }
        elif (self.expencoder == 'clip'):
            return {
                'clip': self.clipmodel.state_dict(),
                'net': self.net.state_dict(),
                'var_sched': self.var_sched.state_dict(),
                'diffusion': self.diffusion.state_dict(),
            }
        elif (self.expencoder == 'dinov2'):
            return {
                'dinov2': self.dinov2model.state_dict(),
                'net': self.net.state_dict(),
                'var_sched': self.var_sched.state_dict(),
                'diffusion': self.diffusion.state_dict(),
            }

    def parameters_to_optimize(self):
        if self.expencoder == 'arcface':
            return [
                {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]
        elif self.expencoder == 'arcfarl':
            return [
                {'params': self.farlmodel.parameters(), 'lr': self.cfg.train.farl_lr},
                {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]
        elif (self.expencoder == 'farl'):
            return [ 
                {'params': self.farlmodel.parameters(), 'lr': self.cfg.train.farl_lr},
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]
        elif (self.expencoder == 'clip'):
            return [ 
                {'params': self.clipmodel.parameters(), 'lr': self.cfg.train.clip_lr},
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]
        elif (self.expencoder == 'dinov2'):
            return [ 
                {'params': self.dinov2model.parameters(), 'lr': self.cfg.train.dinov2_lr},
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]
        else:
            return [
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]

    def encode(self, images, arcface_imgs=None, farl_images=None, clip_images=None, dinov2_images=None):
        codedict = {}
        if self.expencoder == 'arcface':
            codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        elif self.expencoder == 'arcfarl':
            codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
            codedict['farl'] = self.farlmodel.encode_image(farl_images).to(self.device)
        elif self.expencoder == 'farl':
            codedict['farl'] = self.farlmodel.encode_image(farl_images).to(self.device)
        elif self.expencoder == 'clip':
            codedict['clip'] = self.clipmodel.encode_image(clip_images).to(self.device)
        elif self.expencoder == 'dinov2':
            codedict['dinov2'] = self.dinov2model(dinov2_images).to(self.device)

        codedict['images'] = images

        return codedict

    def decode(self, codedict, epoch=0, visualize=False, withpose=False, withexp=False,shapecode=None, expcode=None, rotcode=None, numsamples=1):
        self.epoch = epoch
        pred_theta = None
        e_rand = None
        pred_lmk2d = None
        pred_lmk3d = None
        allcode = 0
        gt_mesh = None
        pred_mesh = None
        gt_flameparam = None
        pred_flameparam = None

        if self.expencoder == 'arcface':
            identity_code = codedict['arcface']
        elif self.expencoder == 'farl':
            identity_code = codedict['farl']
        elif self.expencoder == 'clip':
            identity_code = codedict['clip']
        elif self.expencoder == 'dinov2':
            identity_code = codedict['dinov2']
        elif self.expencoder == 'arcfarl':
            identity_code = torch.cat((codedict['farl'], codedict['arcface']), dim=1)
        batch_size = identity_code.shape[0]

        if (not self.validation) and (not self.testing):
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(batch_size, -1) #:, flame['shape_params'].shape[-1])
            shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]

            ## with expression
            if self.with_exp:
                expcode = flame['exp_params'].view(batch_size, -1) #, flame['exp_params'].shape[-1])
                expcode = expcode.to(self.device)[:, :self.cfg.model.n_exp]

                with torch.no_grad():
                    allcode = expcode
                with torch.no_grad():
                    if self.cfg.net.flame_dim == 53:
                        posecode = flame['pose_params'].view(batch_size, -1)[:,3:6] # flame['pose_params'].shape[-1])
                        allcode = torch.cat([expcode, posecode], dim=1)
                        pred_theta, e_rand, predx0_flameparam = self.diffusion.decode(self.epoch, allcode, identity_code, self.flame, visualize, codedict)

                if visualize:
                    with torch.no_grad(): 
                        pred_pose = predx0_flameparam[:,50:].view(batch_size, -1) #, predx0_flameparam[:,100:].shape[-1])
                        pred_exp = predx0_flameparam[:,:50].view(batch_size, -1)#, predx0_flameparam[:,:100].shape[-1])
                        gt_mesh, lmk2d, lmk3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode) 
                        pred_mesh, pred_lmk2d , pred_lmk3d = self.flame(shape_params=shapecode, expression_params=pred_exp.float(), pose_params=pred_pose.float())
            else:
                with torch.no_grad():
                    allcode = shapecode
                pred_theta, e_rand, predx0_flameparam = self.diffusion.decode(self.epoch, shapecode, identity_code, self.flame, visualize, codedict)
                if visualize:
                    with torch.no_grad():
                        gt_mesh, lmk2d, lmk3d = self.flame(shape_params=shapecode)
                        pred_mesh, pred_lmk2d, pred_lmk3d = self.flame(shape_params=predx0_flameparam.float())


        if self.testing or self.validation:
            with torch.no_grad():
                if self.with_exp:
                    pred_flameparam = self.diffusion.sample(num_points=self.cfg.net.flame_dim, context=identity_code, batch_size=identity_code.shape[0], sampling=self.cfg.model.sampling, shapeparam=shapecode)
                    if self.cfg.net.flame_dim == 53:
                        pred_expparam = pred_flameparam[:,:50]
                        pred_jawparam = pred_flameparam[:,50:] * 0.5
                        pred_mesh, pred_lmk2d, pred_lmk3d = self.flame(shape_params=shapecode, expression_params=pred_expparam.float(), jaw_params=pred_jawparam.float()) 
                else:
                    pred_flameparam = self.diffusion.sample(num_points=self.cfg.net.flame_dim, context=identity_code, batch_size=identity_code.shape[0], sampling=self.cfg.model.sampling, codedict=codedict)
                    pred_mesh, pred_lmk2d, pred_lmk3d = self.flame(shape_params=pred_flameparam.float())

                    if self.validation:
                        flame = codedict['flame']
                        shapecode = flame['shape_params'] 

                        shapecode = flame['shape_params'].view(batch_size, -1) 
                        shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
                        gt_mesh, lmk2d, lmk3d = self.flame(shape_params=shapecode)

        output = {
            'gt_mesh': gt_mesh,
            'gt_flameparam': allcode,
            'pred_mesh': pred_mesh,
            'pred_flameparam': pred_flameparam,
            'pred_theta': pred_theta,
            'e_rand': e_rand,
            'faceid': identity_code,
            'lmk2d': pred_lmk2d,
            'lmk3d': pred_lmk3d,
        }
        return output

    def compute_losses(self, decoder_output, losstype='l1'):
        losses = {}

        pred_theta = decoder_output['pred_theta']
        e_rand = decoder_output['e_rand']
        if losstype == 'mse':
            e_loss = F.mse_loss(pred_theta, e_rand, reduction='mean')
            losses['pred_theta_diff'] = e_loss*100.0
        else:
            e_loss = (pred_theta - e_rand).abs()
            losses['pred_theta_diff'] = torch.mean(e_loss)

        return losses


    def compute_val_losses(self, decoder_output, losstype='l1'):
        losses = {}

        pred_mesh = decoder_output['pred_mesh']
        gt_mesh = decoder_output['gt_mesh']
        mesh_loss = torch.abs(pred_mesh - gt_mesh).mean(dim=(-1,-2))
        losses['pred_mesh_diff'] = torch.mean(mesh_loss)*1e2

        return losses

