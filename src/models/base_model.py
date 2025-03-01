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


from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import sys
from src.configs.config import cfg
from .flame import FLAME


class BaseModel(nn.Module):
    def __init__(self, config=None, device=None, tag=''):
        super(BaseModel, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.tag = tag
        self.use_mask = self.cfg.train.use_mask
        self.device = device
        self.testing = self.cfg.model.testing
        self.validation = self.cfg.model.validation
        self.landmark = self.cfg.model.landmark
        self.uniform = self.cfg.model.uniform
        self.use_reg = self.cfg.model.use_reg
        self.with_exp = self.cfg.model.with_exp
        self.sampling = self.cfg.model.sampling

    def initialize(self):
        self.create_flame(self.cfg.model)
        self.create_model(self.cfg.model)
        self.load_model()
        self.setup_renderer(self.cfg.model)

    def create_flame(self, model_cfg):
        self.flame = FLAME(model_cfg).to(self.device)
        self.average_face = self.flame.v_template.clone()[None]
        self.flame.eval()

    @abstractmethod
    def create_model(self):
        return

    @abstractmethod
    def create_load(self):
        return

    @abstractmethod
    def model_dict(self):
        return

    @abstractmethod
    def parameters_to_optimize(self):
        return

    @abstractmethod
    def encode(self, images, arcface_images):
        return

    @abstractmethod
    def decode(self, codedict, epoch):
        pass

    @abstractmethod
    def compute_losses(self, input, encoder_output, decoder_output):
        pass

    @abstractmethod
    def compute_masks(self, input, decoder_output):
        pass

    def setup_renderer(self, model_cfg):
        self.verts_template_neutral = self.flame.v_template[None]
        self.verts_template = None
        self.verts_template_uv = None


    def create_template(self, B):
        with torch.no_grad():
            if self.verts_template is None:
                self.verts_template_neutral = self.flame.v_template[None]
                pose = torch.zeros(B, self.cfg.model.n_pose, device=self.device)
                pose[:, 3] = 10.0 * np.pi / 180.0  # 48
                self.verts_template, _, _ = self.flame(shape_params=torch.zeros(B, self.cfg.model.n_shape, device=self.device), expression_params=torch.zeros(B, self.cfg.model.n_exp, device=self.device), pose_params=pose)  # use template mesh with open mouth

            if self.verts_template.shape[0] != B:
                self.verts_template_neutral = self.verts_template_neutral[0:1].repeat(B, 1, 1)
                self.verts_template = self.verts_template[0:1].repeat(B, 1, 1)
