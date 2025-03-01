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
from src.models.baselinemodels.flameparamdiffusion_model import FlameParamDiffusionModel
from src.models.network.rank_network import RankMLPNet
from pytorch_lightning import seed_everything
from loguru import logger
import numpy as np

class FlameParamRankModel(BaseModel):
    def __init__(self, config=None, device=None):
        super(FlameParamRankModel, self).__init__(config, device, 'FlameParamRankModel')
        self.expencoder = self.cfg.model.expencoder
        self.preexpencoder = self.cfg.model.preexpencoder
        self.testing = self.cfg.model.testing
        self.validation = self.cfg.model.validation
        self.initialize()
        self.frontfaceindex = np.load(os.path.join(self.cfg.model.pretrained, 'frontindices.npy'))

    def create_model(self, model_config):
        mapping_layers = model_config.mapping_layers
        pretrained_path = None
        if not model_config.use_pretrained:
            pretrained_path = model_config.arcface_pretrained_model
        print("freeze = {}".format(self.cfg.model.with_freeze), flush=True)
        if self.expencoder == 'arcfarl':
            import clip
            self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
            farl_state = torch.load(os.path.join(model_config.pretrained, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth" ))
            self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
            self.farlmodel = self.farlmodel.to(self.device)
            self.arcface = Arcface(pretrained_path=pretrained_path, freeze=self.cfg.model.with_freeze).to(self.device)

        ### Network
        self.net = RankMLPNet(config=self.cfg.net, device=self.device)
        print("in training", flush=True)
        ### IdGen setting for training data to IdRank
        self.precfg = self.cfg.clone()
        self.precfg.train.resume = False
        self.precfg.model.expencoder = self.cfg.model.preexpencoder
        self.precfg.model.nettype = self.cfg.model.prenettype
        self.precfg.net.dims = self.cfg.net.predims
        self.precfg.net.context_dim=512
        self.diffusionmodel = FlameParamDiffusionModel(self.precfg, self.device)

    def load_model(self):
        # DO NOT CHANGE THE ORDER, WE WANT TO START FROM LEARNED ENCODER IF THE TRAINING IS FROM SCRATCH
        print("in load model", flush=True)
        if self.cfg.train.resumepretrain:
            model_path = os.path.join(self.cfg.train.resume_pretrain_checkpoint)
            if os.path.exists(model_path):
                logger.info(f'[{self.tag}] Pretrained Trained model found. Path: {model_path} | GPU: {self.device}')
                checkpoint = torch.load(model_path)
                self.diffusionmodel.load_for_test(model_path)
                if not self.cfg.train.resume:
                    print("no resume, load from pretrained model", flush=True)
                    self.arcface.load_state_dict(checkpoint['arcface'])
                    if self.cfg.model.preexpencoder == 'arcfarl':
                        self.farlmodel.load_state_dict(checkpoint['farl'])
            else:
                print("model path = ", model_path)
                print("no model exists")
                exit()
        if self.cfg.train.resume:
            print("resuming checkpoint", flush=True)
            model_path = os.path.join(self.cfg.train.resume_checkpoint)
            if os.path.exists(model_path):
                logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
                checkpoint = torch.load(model_path)
                if 'net' in checkpoint:
                    print("rank net")
                    self.net.load_state_dict(checkpoint['net'])
                if 'arcface' in checkpoint:
                    print("rank arcface")
                    self.arcface.load_state_dict(checkpoint['arcface'])
                if 'farl' in checkpoint:
                    print("rank farl")
                    self.farlmodel.load_state_dict(checkpoint['farl'])
                del checkpoint
            else:
                logger.info(f'[{self.tag}] Checkpoint {model_path} not available starting from scratch!')
                exit()

    def model_dict(self):
        if self.expencoder == 'arcfarl':
            return {
                    'net': self.net.state_dict(),
                    'arcface': self.arcface.state_dict(),
                    'farl': self.farlmodel.state_dict()
            }
        elif self.expencoder == 'arcface':
            return {
                    'net': self.net.state_dict(),
                    'arcface': self.arcface.state_dict(),
            }
        else:
            return {
                    'net': self.net.state_dict(),
            }

    def parameters_to_optimize(self):
        if self.expencoder == 'arcfarl':
            print("optimizing arcfarl", flush=True)
            return [
                {'params': self.net.parameters(), 'lr': self.cfg.train.rank_lr},
                {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
                {'params': self.farlmodel.parameters(), 'lr': self.cfg.train.farl_lr},
            ]
        elif self.expencoder == 'arcface':
            return [
                {'params': self.net.parameters(), 'lr': self.cfg.train.rank_lr},
                {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
            ]
        else:
            return [
                {'params': self.net.parameters(), 'lr': self.cfg.train.rank_lr},
            ]


    def encode(self, images, arcface_imgs, farl_images):
        codedict1 = {}
        codedict2 = {}
        if self.expencoder == 'arcfarl': 
            codedict1['arcface'] = F.normalize(self.arcface(arcface_imgs))
            codedict1['farl'] = self.farlmodel.encode_image(farl_images).to(self.device)
        elif self.expencoder == 'arcface':
            codedict1['arcface'] = F.normalize(self.arcface(arcface_imgs))

        if not self.testing:
            with torch.no_grad():
                codedict2 = self.diffusionmodel.encode(images, arcface_imgs, farl_images)
            return codedict1, codedict2
        else:
            return codedict1, None

    def decode(self, codedict, codedictpre, epoch=0, visualize=False, withflame=False, numsamples = 1, withpose=False, withexp=False,shapecode=None, expcode=None, posecode=None, fixed_noise=None):
        self.epoch = epoch
        pred_theta = None
        e_rand = None
        gt_mesh = None
        pred_mesh = None
        pred_lmk2d = None
        pred_lmk3d = None
        allcode = 0
        flameparam_x0 = torch.tensor(0).float().to(self.device)
        self.numsamples = numsamples

        if self.expencoder == 'arcfarl':
            identity_code = torch.cat((codedict['farl'], codedict['arcface']), dim=1)
            with torch.no_grad():
                if self.preexpencoder == 'arcface':
                    identity_code_pre = codedictpre['arcface']
                else:
                    identity_code_pre = torch.cat((codedictpre['farl'], codedictpre['arcface']), dim=1)
        elif self.expencoder == 'arcface':
            identity_code = codedict['arcface'] #torch.cat((codedict['farl'], codedict['arcface']), dim=1)
            with torch.no_grad():
                if self.preexpencoder == 'arcface':
                    identity_code_pre = codedictpre['arcface']
                else:
                    identity_code_pre = torch.cat((codedictpre['farl'], codedictpre['arcface']), dim=1)

        batch_size = identity_code.shape[0]
        if numsamples > 1:
            identity_code_one = identity_code.clone()
            identity_code = identity_code.tile(1,1,numsamples).view(identity_code.shape[0]*numsamples, -1)
            with torch.no_grad():
                identity_code_pre = identity_code_pre.tile(1,1,numsamples).view(identity_code_pre.shape[0]*numsamples, -1)

        pred_mesh_full = None
        residualpred = None
        if self.training or self.validation:
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(batch_size, -1) 
            shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            batch_size = identity_code.shape[0]

            with torch.no_grad():
                gt_mesh, lmk2d, lmk3d = self.flame(shape_params=shapecode) 
                gt_mesh_full = gt_mesh.clone()
                gt_mesh = gt_mesh[:,self.frontfaceindex]

            if numsamples > 1:
                shapecode = shapecode.tile(1,1,numsamples).view(batch_size, -1)

        with torch.no_grad():
            batch_size = identity_code.shape[0]
            pred_flameparam = self.diffusionmodel.diffusion.sample(num_points=self.cfg.net.flame_dim, context=identity_code_pre, batch_size=batch_size, sampling=self.cfg.model.sampling)
            if withexp:
                pred_expparam = pred_flameparam[:,:100]
                if withpose:
                    pred_poseparam = pred_flameparam[:,100:]
                    pred_mesh, pred_lmk2d, pred_lmk3d = self.flame(shape_params=shapecode, expression_params=pred_expparam.float(), pose_params=pred_poseparam.float()) 
                else:
                    pred_mesh, pred_lmk2d, pred_lmk3d = self.flame(shape_params=shapecode, expression_params=pred_expparam.float()) 
            else:
                pred_mesh, pred_lmk2d, pred_lmk3d = self.flame(shape_params=pred_flameparam.float())

            pred_mesh_full = pred_mesh.clone()
            pred_mesh = pred_mesh.view(-1, numsamples, 5023, 3)
            pred_mesh = 1e3*pred_mesh[:,:,self.frontfaceindex] 
            
            meanpred = torch.mean(pred_mesh, dim=1)
            meanpred = meanpred.tile(1,self.numsamples,1).reshape(meanpred.shape[0], numsamples, meanpred.shape[-2], meanpred.shape[-1])
            residualpred = (pred_mesh - meanpred)
            mean_residual = torch.cat((meanpred, residualpred),dim=2)

        smax = None
        if not self.training:
            with torch.no_grad():
                print("validating")
                smax, pred_vertex_error= self.net(x=mean_residual, context=identity_code_one, numsamples = self.numsamples)
        else:
            smax, pred_vertex_error= self.net(x=mean_residual, context=identity_code_one, numsamples = self.numsamples)
       
        gt_mesh = 1e3*gt_mesh.tile(1,numsamples,1).reshape(gt_mesh.shape[0],numsamples,-1,3)
        identity_code = identity_code.view(-1,self.numsamples,identity_code.shape[1])
        identity_code = identity_code[:,0]
        output = {
            'gt_mesh': gt_mesh,
            'pred_mesh': pred_mesh,
            'residual_pred_mesh': residualpred,
            'gt_mesh_full': gt_mesh_full,
            'pred_mesh_full': pred_mesh_full,
            'gt_flameparam': allcode,
            'pred_flameparam': pred_flameparam,
            'gt_flameparam': allcode,
            'pred_vertex_error': pred_vertex_error,
            'softmax_value': smax,
            'identity_code': identity_code
        }

        return output
 
    def decodetest(self, codedict, pred_meshes, epoch=0, visualize=False, withflame=False, numsamples = 1, withpose=False, withexp=False,shapecode=None, expcode=None, posecode=None, fixed_noise=None):
        self.epoch = epoch
        allcode = 0
        self.numsamples = numsamples

        if self.expencoder == 'arcfarl':
            identity_code = torch.cat((codedict['farl'], codedict['arcface']), dim=1)
        elif self.expencoder == 'arcface':
            identity_code = codedict['arcface']

        batch_size = identity_code.shape[0]

        pred_mesh_full = pred_meshes.clone()
        pred_mesh = (pred_meshes[:,self.frontfaceindex]).unsqueeze(0)
        batch_size = identity_code.shape[0]

        with torch.no_grad():
            print("validating")
            meanpred = torch.mean(pred_mesh, dim=1)
            meanpred = meanpred.tile(1,self.numsamples,1).reshape(batch_size, numsamples, meanpred.shape[-2], meanpred.shape[-1])
            residualpred = (pred_mesh - meanpred)
            final_mesh  = torch.cat((meanpred, residualpred),dim=2)
            smax, pred_vertex_error= self.net(x=final_mesh, context=identity_code, numsamples = self.numsamples)

        output = {
            'pred_mesh': pred_mesh,
            'pred_mesh_full': pred_mesh_full,
            'pred_vertex_error': pred_vertex_error,
            'softmax_value': smax
        }

        return output

    def compute_losses(self, decoder_output, visualize):
        losses = {}
        loss = None
        pred_vertex_error = decoder_output['pred_vertex_error']
        pred_mesh = decoder_output['pred_mesh']
        gt_mesh = decoder_output['gt_mesh']
        smax = decoder_output['softmax_value']
        if self.cfg.net.losstype == 'Softmaxloss':
            print("softmax loss", flush=True)
            loss = self.compute_Softmaxloss(gt_mesh, pred_mesh, pred_vertex_error, smax, visualize)
        elif self.cfg.net.losstype == 'BCElistnetloss':
            loss = self.compute_BCElistnetloss(gt_mesh, pred_mesh, pred_vertex_error, visualize)
        elif self.cfg.net.losstype == 'Softmaxlistnetloss':
            loss = self.compute_Softmaxlistnetloss(gt_mesh, pred_mesh, pred_vertex_error, visualize)

        return loss


    def compute_Softmaxloss(self, gt, samples, predscore, smax, visualize):
        batch_size = samples.shape[0]
        splitsize = int(batch_size / self.numsamples)
        with torch.no_grad():
            gtloss = (torch.abs(gt - 1000*samples)).sum(dim=(1,2))
            gtloss = gtloss.reshape(splitsize, -1)
            gtonehot = torch.zeros_like(gtloss)
            minmaxindex = 0
            if 'reverse' in self.cfg.net.rankarch:
                minmaxindex = torch.argmax(gtloss, dim=1)
                gtonehot.scatter_(1, minmaxindex.unsqueeze(1), torch.ones_like(gtloss))
            else:
                minmaxindex = torch.argmin(gtloss, dim=1)
                gtonehot.scatter_(1, minmaxindex.unsqueeze(1), torch.ones_like(gtloss))

        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        rankloss = loss(predscore, gtonehot)
        loss = {'rankloss': rankloss}
        return loss


    def compute_BCElistnetloss(self, gt, samples, predscore, visualize):
        batch_size = samples.shape[0]
        with torch.no_grad():
            gtloss = (torch.abs(gt - samples)).mean(dim=(-2,-1))
            gtonehot = torch.zeros_like(gtloss).to(self.device)
            minmaxindex = torch.argmin(gtloss, dim=1)
            gtonehot.scatter_(1, minmaxindex.unsqueeze(1).to(self.device), torch.ones_like(gtloss).to(self.device))

        loss = torch.nn.BCELoss(reduction='mean')
        sig = torch.nn.Sigmoid()
        rankloss = 1e3*loss(sig(predscore), gtonehot)
        if visualize:
            print("rankloss =", rankloss, flush=True)
        loss = {'rankloss': rankloss}
        return loss

    def compute_Softmaxlistnetloss(self, gt, samples, predscore, visualize):
        batch_size = samples.shape[0]
        with torch.no_grad():
            gtloss = (torch.abs(gt - samples)).mean(dim=(-2,-1))
            gtlossneg = -gtloss
            softmax = torch.nn.Softmax(dim=-1)
            gtlosssoftmax = softmax(gtlossneg).to(self.device)

        loss = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-200)
        rankloss = loss(predscore, gtlosssoftmax)
        if visualize:
            print("rankloss =", rankloss, flush=True)
        loss = {'rankloss': rankloss}
        return loss
