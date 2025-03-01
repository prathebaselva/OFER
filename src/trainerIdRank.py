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
# Contact: mica@tue.mpg.de


import os
import random
import sys
import math
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.datasets as datasets
from src.configs.config import cfg
from src.utils import util
import trimesh
import shutil

sys.path.append("./src")
from pytorch_lightning import seed_everything


def print_info(rank):
    props = torch.cuda.get_device_properties(rank)

    logger.info(f'[INFO]            {torch.cuda.get_device_name(rank)}')
    logger.info(f'[INFO] Rank:      {str(rank)}')
    logger.info(f'[INFO] Memory:    {round(props.total_memory / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Allocated: {round(torch.cuda.memory_allocated(rank) / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Cached:    {round(torch.cuda.memory_reserved(rank) / 1024 ** 3, 1)} GB')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainerIdRank(object):
    def __init__(self, model, pretrainedmodel=None, config=None, pretrainconfig=None,device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.cfg2 = pretrainconfig
        self.seed = self.cfg.seed
        if self.cfg.withseed:
            seed_everything(self.seed)
        self.numsamples = self.cfg.model.numsamples

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
        self.withval = self.cfg.model.with_val
        print(self.cfg, flush=True)
        self.epoch = 0
        self.global_step = 0

        print(self.withval, flush=True)

        # autoencoder model
        self.model = model.to(self.device)
        self.faces = model.flame.faces_tensor.cpu().numpy()
        self.configure_optimizers()
        if self.cfg.train.resume:
            self.load_checkpoint()


        # reset optimizer if loaded from pretrained model
        if self.cfg.train.reset_optimizer:
            self.configure_optimizers()  # reset optimizer
            logger.info(f"[TRAINER] Optimizer was reset")

        if self.cfg.train.write_summary:
            print("in write summary ", flush=True)
            from torch.utils.tensorboard import SummaryWriter
            logdir = os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, self.cfg.net.tag)
            os.makedirs(logdir, exist_ok=True)
            print("logdir =", logdir, flush=True)
            self.writer = SummaryWriter(log_dir=logdir)
            print(self.writer, flush=True)

        print_info(device)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            weight_decay=self.cfg.train.weight_decay,
            params=self.model.parameters_to_optimize(),
            amsgrad=False)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                         self.optimizer,
                         factor=0.99,
                         patience=2,
                         mode='min',
                         threshold=1e-4,
                         eps=0,
                         min_lr=0)

    def load_checkpoint(self):
        self.epoch = 0
        self.global_step = 0
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        model_path = os.path.join(self.cfg.checkpoint_dir, 'model_best.tar')

        if os.path.exists(self.cfg.train.resume_checkpoint):
            model_path = os.path.join(self.cfg.train.resume_checkpoint)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f"[TRAINER] Resume training from {model_path}")
            logger.info(f"[TRAINER] Start from step {self.global_step}")
            logger.info(f"[TRAINER] Start from epoch {self.epoch}")
        else:
            logger.info('[TRAINER] Model path not found, start training from scratch')

    def save_checkpoint(self, filename):
        if self.device == 0:
            model_dict = self.model.model_dict()

            model_dict['optimizer'] = self.optimizer.state_dict()
            model_dict['scheduler'] = self.scheduler.state_dict()
            if self.withval:
                model_dict['validator'] = self.validator.state_dict()
            model_dict['epoch'] = self.epoch
            model_dict['global_step'] = self.global_step
            model_dict['batch_size'] = self.batch_size

            torch.save(model_dict, filename)

    def training_step(self, batch, visualize=False):
        self.model.train()
        self.model.validation = False
        self.model.istesting = False

        #Original images
        images = batch['image'].to(self.device)
        farlimages = batch['farl'].to(self.device)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        farlimages = farlimages.view(-1, farlimages.shape[-3], farlimages.shape[-2], farlimages.shape[-1])
        # Flame parameters
        flame = batch['flame']
        exp = batch['exp']
        pose = batch['pose']
        pose_valid = batch['pose_valid']
        lmk = batch['lmk']
        lmk_valid = batch['lmk_valid']
        currpredmesh = batch['currpredmesh']
        bestallpredmesh = batch['bestallpredmesh']
        actorpredmesh = batch['actorpredmesh']
        actorpredflame = batch['actorpredflame']
        gtmesh = batch['gtmesh']

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
            encoder_output, encoder_output_pre = self.model.encode(images, arcface, farlimages)
        encoder_output['flame'] = flame
        encoder_output['exp'] = exp
        encoder_output['pose'] = pose
        encoder_output['lmk'] = lmk
        encoder_output['pose_valid'] = pose_valid
        encoder_output['lmk_valid'] = lmk_valid
        encoder_output['currpredmesh'] = currpredmesh
        encoder_output['bestallpredmesh'] = bestallpredmesh
        encoder_output['actorpredmesh'] = actorpredmesh
        encoder_output['actorpredflame'] = actorpredflame
        encoder_output['gtmesh'] = gtmesh
        encoder_output_pre['flame'] = flame

        decoder_output = self.model.decode(encoder_output, encoder_output_pre, self.epoch, visualize=visualize, numsamples=self.numsamples)
        losses = self.model.compute_losses(decoder_output, visualize)

        all_loss = 0.
        vertex_loss = 0.
        losses_key = losses.keys()

        rank_loss = losses['rankloss']
        losses = {}
        losses['all_loss'] = rank_loss

        opdict = \
            {
                'images': images,
            }

        if 'gt_mesh' in decoder_output:
            opdict['gt_mesh']= decoder_output['gt_mesh']
        if 'pred_mesh' in decoder_output:
            opdict['pred_mesh']= decoder_output['pred_mesh']
        if 'gt_flameparam' in decoder_output:
            opdict['gt_flameparam']= decoder_output['gt_flameparam']
        if 'pred_flameparam' in decoder_output:
            opdict['pred_flameparam']= decoder_output['pred_flameparam']
        return losses, opdict

    def validation_step(self):
        self.model.eval()
        self.model.istesting = True
        return self.validator.run()

    def evaluation_step(self):
        pass

    def prepare_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.cfg.model.flametype == 'flame20':
            self.train_dataset, total_images = datasets.build_flame_train(self.cfg.dataset, self.cfg.model.expencoder, self.device)
        elif self.cfg.model.flametype == 'flame23':
            self.train_dataset, total_images = datasets.build_flame_train_23(self.cfg.dataset, self.cfg.model.expencoder, self.device)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator)
        self.train_iter = iter(self.train_dataloader)

        logger.info(f'[TRAINER] Training  dataset is ready with {len(self.train_dataset)} actors and {total_images} images.')

    def run(self):
        self.prepare_data()
        iters_every_epoch = math.ceil(len(self.train_dataset)/ self.batch_size)
        start_epoch = self.epoch
        max_epochs = self.cfg.train.max_epochs
        self.train_best_loss = np.Inf
        self.val_best_loss = np.Inf
        for epoch in range(start_epoch, max_epochs):
            epochvalloss = 0
            epochvalcount = 0
            epochtrainloss = 0
            epochtraincount = 0
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch + 1}/{max_epochs}]"):
                if self.global_step > self.cfg.train.max_steps:
                    break
                try:
                    #print("in try", flush=True)
                    batch = next(self.train_iter)
                except Exception as e:
                    #logger.info(f'in exception train')
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                visualizeTraining = self.global_step % self.cfg.train.vis_steps == 0

                self.optimizer.zero_grad()
                batch_size = batch['batchsize'].sum().item()
                losses, opdict = self.training_step(batch, visualize=visualizeTraining)
                all_loss = losses['all_loss']
                #if not self.withval:
                epochtrainloss += (batch_size * all_loss.item())
                epochtraincount += batch_size

                all_loss.backward()
                self.optimizer.step()

                if self.global_step % self.cfg.train.log_steps == 0 and self.device == 0:
                    loss_info = f"\n" \
                                f"  Epoch: {epoch}\n" \
                                f"  Step: {self.global_step}\n" \
                                f"  Iter: {step}/{iters_every_epoch}\n" \
                                f"  Rank LR: {self.optimizer.param_groups[0]['lr']}\n" \
                                f"  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                    if self.cfg.model.expencoder == 'arcfarl':
                        loss_info = f"\n" \
                                    f"  Epoch: {epoch}\n" \
                                    f"  Step: {self.global_step}\n" \
                                    f"  Iter: {step}/{iters_every_epoch}\n" \
                                    f"  Rank LR: {self.optimizer.param_groups[0]['lr']}\n" \
                                    f"  Farl LR: {self.optimizer.param_groups[1]['lr']}\n" \
                                    f"  Arcface LR: {self.optimizer.param_groups[2]['lr']}\n" \
                                    f"  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                    logger.info(loss_info)

                if (self.global_step > 1000) and self.global_step % self.cfg.train.checkpoint_epochs_steps == 0:
                    os.makedirs(self.cfg.output_dir, exist_ok=True)
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model_'+str(self.cfg.net.tag)+'.tar'))

                self.global_step += 1

            if self.withval:
                logger.info("validation")
                val_loss, val_batch_size = self.validation_step()
                epochvalloss = val_loss/val_batch_size

                logger.info(f'epochvalloss = {epochvalloss}')
                if self.cfg.train.write_summary:
                    self.writer.add_scalar('Loss/val', epochvalloss, epoch)
                if epochvalloss <= self.val_best_loss:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'best_models', 'model_val_'+str(self.cfg.net.tag)+'_best.tar'))
                    self.val_best_loss = epochvalloss

            epochtrainloss = epochtrainloss / epochtraincount
            if self.cfg.train.write_summary:
                self.writer.add_scalar('Loss/train', epochtrainloss, epoch)

            if epochtrainloss < self.train_best_loss:
                logger.info(f'best train {epoch}')
                logger.info(f'{epochtrainloss}')
                self.train_best_loss = epochtrainloss
                os.makedirs(os.path.join(self.cfg.output_dir, 'best_models'), exist_ok=True)
                self.save_checkpoint(os.path.join(self.cfg.output_dir, 'best_models', 'model_train_'+str(self.cfg.net.tag)+'_best.tar'))
            if epoch >= 650:
                print("trained for 650 epochs")
                logger.info(f'[TRAINER] Fitting has ended!')
                exit()


            if self.withval:
                self.scheduler.step(epochvalloss)
            else:
                self.scheduler.step(epochtrainloss)
            self.epoch += 1

        self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))
        logger.info(f'[TRAINER] Fitting has ended!')
