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
import subprocess
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader

import datasets
from utils import util
from utils.best_model import BestModel
import random
import math

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() %2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Validator(object):
    def __init__(self, trainer):
        logger.info('in validator')
        self.trainer = trainer
        self.device = self.trainer.device
        self.model = self.trainer.model
        self.model.validation = True
        self.batch_size = self.trainer.cfg.dataset.batch_size
        self.cfg = deepcopy(self.trainer.cfg)
        self.seed = self.cfg.seed
        self.cfg.model.validation = True
        self.cfg.model.sampling = 'ddpm'
        self.device = trainer.device

        self.embeddings = {}
        self.best_model = BestModel(trainer)
        self.prepare_data(self.model.with_exp)

    def prepare_data(self, with_exp=False):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        logger.info(f'Preparing validator data')

        self.val_dataset, total_images = datasets.build_flame_val_23(self.cfg.dataset, self.cfg.model.expencoder, self.device)

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator)

        self.val_iter = iter(self.val_dataloader)
        logger.info(f'[VALIDATOR] Validation dataset is ready with {len(self.val_dataset)} actors and {total_images} images.')

    def state_dict(self):
        return {
            'embeddings': self.embeddings,
            'best_model': self.best_model.state_dict(),
        }

    def load_state_dict(self, dict):
        self.embeddings = dict['embeddings']
        self.best_model.load_state_dict(dict['best_model'])

    def update_embeddings(self, actors, arcface):
        B = len(actors)
        for i in range(B):
            actor = actors[i]
            if actor not in self.embeddings:
                self.embeddings[actor] = []
            self.embeddings[actor].append(arcface[i].data.cpu().numpy())

    def run(self):
        with torch.no_grad():
            # In the case of using multiple GPUs
            if self.trainer.device != 0:
                logger.info("trainer device = {self.trainer.device}")
                return

            self.model.eval()
            self.model.validation = True
            self.model.testing = True
            optdicts = []
            allbatchsize = 0
            iters_every_epoch = math.ceil(len(self.val_dataset)/ self.batch_size)
            for step in tqdm(range(iters_every_epoch)):
                try:
                    batch = next(self.val_iter)
                except Exception as e:
                    logger.info(f'in exception')
                    logger.info(f'{e}')
                    self.val_iter = iter(self.val_dataloader)
                    batch = next(self.val_iter)

                batch_size = batch['batchsize'].sum().item()
                    #break
                #Original images
                images = batch['image'].to(self.device)
                farlimages = batch['farl'].to(self.device)
                images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                farlimages = farlimages.view(-1, farlimages.shape[-3], farlimages.shape[-2], farlimages.shape[-1])
                flame = batch['flame']
                exp = batch['exp']
                pose = batch['pose']
                pose_valid = batch['pose_valid']
                lmk = batch['lmk']
                lmk_valid = batch['lmk_valid']

                # arcface images
                arcface = batch['arcface']
                arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device)

                inputs = {
                    'images': images,
                    'farlimages': farlimages,
                    'dataset': batch['dataset'][0]
                }
               
                if self.cfg.net.context_dim == 0:
                    encoder_output = {}
                    encoder_output['arcface'] = None
                    encoder_output['images'] = None
                else:
                    encoder_output = self.model.encode(images, arcface, farlimages)
                encoder_output['flame'] = flame
                encoder_output['exp'] = exp
                encoder_output['pose'] = pose
                encoder_output['lmk'] = lmk
                encoder_output['pose_valid'] = pose_valid
                encoder_output['lmk_valid'] = lmk_valid


                visualize = step % 100 == 0
                decoder_output = self.model.decode(encoder_output, 0, visualize=visualize)
                losses = self.model.compute_val_losses(decoder_output, visualize)

                all_loss = 0.
                vertex_loss = 0.
                loss = losses['pred_mesh_diff'] #sum(losses.values())
                optdicts.append(loss)
                allbatchsize += batch_size 

            # Calculate averages
            weighted_average = 0.
            average = 0.
            avg_per_dataset = {}
            totalloss = torch.tensor(optdicts).sum()
            average = totalloss.item() / allbatchsize

            loss_info = f"Step: {self.trainer.global_step},  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
            loss_info += f'  validation loss (average)         : {average:.5f} \n'
            logger.info(loss_info)

            self.trainer.writer.add_scalar('val/average', average, global_step=self.trainer.global_step)

            # Save best model
            smoothed_weighted, smoothed = self.best_model(weighted_average, average, self.cfg.net.tag)
            self.trainer.writer.add_scalar(f'val/smoothed_average', smoothed, global_step=self.trainer.global_step)
            return totalloss, allbatchsize

