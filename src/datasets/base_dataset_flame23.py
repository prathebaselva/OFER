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
import re
from abc import ABC
from functools import reduce
from pathlib import Path
import cv2
import clip
import glob

import loguru
import numpy as np
import torch
import trimesh
import scipy.io
from loguru import logger
from skimage.io import imread
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from src.models.flame import FLAME


class BaseDatasetFlame23(Dataset, ABC):
    def __init__(self, name, config, device, isEval):
        self.n_images = config.n_images
        self.occlusion = config.occlusion
        self.isEval = isEval
        self.n_train = np.Inf
        self.imagepaths = []
        self.lmk = config.lmk
        self.face_dict = {}
        self.name = name
        self.device = device
        self.min_max_K = 0
        self.cluster = False
        self.dataset_root = config.root
        self.total_images = 0
        self.config = config
        self.pretrained = config.pretrained
        self.tocenter = config.tocenter
        self.flipchannels = config.flipchannels

        self.flame_folder = 'FLAME23_parameters'
        self.farlmodel, self.farlpreprocess = clip.load("ViT-B/16", device="cpu")
        self.clipmodel, self.clippreprocess = clip.load("ViT-B/32", device="cpu")

        self.dinotransform  = T.Compose([
            T.ToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.Normalize(mean=[0.5],std=[0.5]),])
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        image_list = os.path.join(self.dataset_root, 'image_paths/arcface23', self.name+'.npy')
        self.allpredmesh = []
        self.allpredflame = []
        print(self.name, flush=True)

        logger.info(f'[{self.name}] Load cached file list: ' + image_list)
        self.face_dict = np.load(image_list, allow_pickle=True).item()
        self.flame = FLAME(self.config).to(self.device)
        self.imagepaths = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.imagepaths)} actors loaded!')

        arcface_input = 'arcface_input'
        clip_input = 'clip_input'

        self.image_folder = arcface_input
        self.set_smallest_numimages()

    def set_smallest_numimages(self):
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][1])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        self.total_images = reduce(lambda k, l: l + k, map(lambda e: len(self.face_dict[e][1]), self.imagepaths))
        loguru.logger.info(f'Dataset {self.name} with min num of images = {self.min_max_K} max num of images = {max_min_k} length = {len(self.face_dict)} total images = {self.total_images}')
        return self.min_max_K

    def compose_transforms(self, *args):
        self.transforms = T.Compose([t for t in args])

    def get_arcface_path(self, image_path):
        return re.sub('png|jpg', 'npy', str(image_path))

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, index):
        actor = self.imagepaths[index]
        params_path, images = self.face_dict[actor]

        if 'STIRLING_FRONTFACE_HQ' in self.name or 'STIRLING_FRONTFACE_LQ' in self.name:
            data = self.name.split('_')
            name = data[0]+'_'+data[1]
            qual = data[2]
            images = [Path(self.dataset_root, name, self.image_folder, path) for path in images]
            predmeshfilepath = (os.path.join(self.predmeshfolder, actor, '*.ply'))
            allpredmeshfile = glob.glob(predmeshfilepath.replace('frontface','FRONTFACE'))
            flamefilepath = (os.path.join(self.predmeshfolder, actor, '*flame.npy'))
            allpredflamefile = glob.glob(flamefilepath.replace('frontface','FRONTFACE'))
            self.actorpredmesh = []
            self.gtmesh = []
            self.actorpredflame = []
            self.gtflame = []
            for docs in allpredmeshfile:
                mesh = trimesh.load(docs)
                meshv = np.array(mesh.vertices)
                if "gt.ply" in docs:
                    self.gtmesh.append(torch.Tensor(meshv))
                    continue
                self.actorpredmesh.append(torch.Tensor(meshv))
            for docs in allpredflamefile:
                pflame = np.load(docs)
                if "gtflame.npy" in docs:
                    self.gtflame.append(torch.Tensor(pflame))
                    #print(docs, flush=True)
                    continue
                self.actorpredflame.append(torch.Tensor(pflame))
        else:
            if "TRAIN" in self.name or "VAL" in self.name:
                data = self.name.split("_")
                self.name = data[0]
            images = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images]

        sample_list = np.array(np.random.choice(range(len(images)), size=self.n_images, replace=True))

        K = self.n_images
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images))[:K])

        if self.name == 'STIRLING': 
            foldername = ''
        else:
            foldername = ''

        if self.name == 'TEMPEH':
            params = np.load(os.path.join(self.dataset_root,self.name, params_path), allow_pickle=True)
            shape_param = torch.tensor(params['shape']).float().to(self.device)
            exp_param = torch.tensor(params['exp']).float().to(self.device)
            pose_param = torch.tensor(params['pose']).float().to(self.device)
            neck_param = torch.tensor(params['pose'][:3]).float().to(self.device)
            jaw_param = torch.tensor(params['pose'][3:6]).float().to(self.device)
            eye_param = torch.tensor(params['pose'][6:]).float().to(self.device)
            trans_param = torch.tensor(params['trans']).float().to(self.device)

            flame = {
                'shape_params': torch.cat(K * [shape_param], dim=0),
                'exp_params': torch.cat(K * [exp_param], dim=0),
                'pose_params': torch.cat(K * [pose_param], dim=0),
                'eye_params': torch.cat(K * [eye_param], dim=0),
                'jaw_params': torch.cat(K * [jaw_param], dim=0),
                'neck_params': torch.cat(K * [neck_param], dim=0),
                'trans_params': torch.cat(K * [trans_param], dim=0),
            }
            exp = torch.ones(K)

        elif self.name == 'COMA':
            params = np.load(os.path.join(self.dataset_root, self.name, params_path), allow_pickle=True).item()
            shape_param = torch.tensor(params['shape']).float().to(self.device)
            exp_param = torch.tensor(params['exp']).float().to(self.device)
            jaw_param = torch.tensor(params['jaw']).float().to(self.device)
            rot_param = torch.tensor(params['rot']).float().to(self.device)
            eye_param = torch.tensor(params['eye']).float().to(self.device)
            neck_param = torch.tensor(params['neck']).float().to(self.device)
            trans_param = torch.tensor(params['trans']).float().to(self.device)

            with torch.no_grad():
                flame_verts_shape,_,_ = self.flame(shape_params=shape_param, expression_params=exp_param,
                        rot_params=rot_param, jaw_params=jaw_param, eye_pose_params=eye_param, neck_pose_params=neck_param,
                        )
                flame_verts_shape = flame_verts_shape.squeeze()
                self.faces = self.flame.faces_tensor.cpu().numpy()
            # Each images share the shape flame parameter of the actor
            # Thus multiply the parameters by the number of images = K
            flame = {
                'shape_params': torch.cat(K * [shape_param], dim=0),
                'exp_params': torch.cat(K * [exp_param], dim=0),
                'jaw_params': torch.cat(K * [jaw_param], dim=0),
                'rot_params': torch.cat(K * [rot_param], dim=0),
                'eye_params': torch.cat(K * [eye_param], dim=0),
                'neck_params': torch.cat(K * [neck_param], dim=0),
                'trans_params': torch.cat(K * [trans_param], dim=0),
            }
            exp = torch.ones(K)
            #exit()
        elif self.name == 'AFLW2000':
            params = scipy.io.loadmat(os.path.join(self.dataset_root, self.name, self.flame_folder, params_path))
            shape_param = torch.Tensor(params['Shape_Para']).to(self.device)
            exp_param = torch.Tensor(params['Exp_Para']).to(self.device)
            shape_param = shape_param.squeeze()[None]
            exp_param = exp_param.squeeze()[None]
            with torch.no_grad():
                flame_verts_shape,_,_ = self.flame_withexp(shape_params=shape_param, expression_params=exp_param)
                logger.info(flame_verts_shape.shape)
                flame_verts_shape = flame_verts_shape.squeeze()
                center = (torch.max(flame_verts_shape, 0)[0] + torch.min(flame_verts_shape,0)[0])/2.0
                gt_verts = (flame_verts_shape - center).float()
            flame = {
                'shape_params': torch.cat(K * [gt_verts[None]], dim=0)
            }
            exp = torch.ones(K)
        else:
            if 'STIRLING' in self.name: # == 'STIRLING_HQ' or self.name == 'STIRLING_LQ':
                name = 'STIRLING'
                floc = os.path.join(self.dataset_root, name, self.flame_folder, foldername, params_path)
                params = np.load(floc, allow_pickle=True)
            else:
                if "TRAIN" in self.name or "VAL" in self.name:
                    data = self.name.split('_')
                    self.name = data[0]
                    qual = data[1]
                params = np.load(os.path.join(self.dataset_root, self.name, self.flame_folder, foldername, params_path), allow_pickle=True)
            shape_param = torch.tensor(params['shape'][None]).to(self.device)

            # Each images share the shape flame parameter of the actor
            # Thus multiply the parameters by the number of images = K
            flame = {
                'shape_params': torch.cat(K * [shape_param], dim=0),
            }
            exp = torch.zeros(K)

        images_list = []
        imagesfarl_list = []
        arcface_list = []
        clip_list = []
        dinov2_list = []
        pose_list = []
        lmk_list = []
        pose_valid_list = []
        lmk_valid_list = []
        currpredmesh_list = []
        imagenames = []
        for i in sample_list:
            image_path = images[i]
            image_name = str(image_path).split('/')[-1]
            imagebasepath = str(image_path)[:-len(image_name)]
            image_name = image_name[:-4]
            if 'FRONTFACE' in self.name:
                predmeshfile = glob.glob(os.path.join(self.bestpredmeshfolder, actor, image_name+'*.ply'))
                mesh = trimesh.load(predmeshfile[0])
                predmeshverts = np.array(mesh.vertices)
                currpredmesh_list.append(torch.Tensor(predmeshverts))

            imagenames.append(self.name+'_'+actor+'_'+image_name)
            if os.path.exists(image_path):
            
                imagefarl = self.farlpreprocess(Image.open(image_path))
                image = np.array(imread(image_path), dtype=np.float32)
                image = image / 255.
                arcface_path = self.get_arcface_path(image_path)
                arcface_image = np.load(self.get_arcface_path(image_path), allow_pickle=True)
                if self.flipchannels:
                    arcface_image = arcface_image[[2,1,0],:,:]
               
                clip_image = self.clippreprocess(Image.open(image_path)).unsqueeze(0)

                dinov2_image = self.dinotransform(Image.open(image_path))[:3].unsqueeze(0)
                pose_path = os.path.join(imagebasepath, str(image_name)+'_pose.npy')
                if os.path.exists(pose_path):
                    pose = np.load(pose_path, allow_pickle=True)
                    if len(pose) == 3:
                        pose = pose[0]
                    elif len(pose) == 2:
                        pose = pose[0][0]

                    if not isinstance(pose, np.ndarray):
                        pose = np.zeros(3)
                        pose_valid_list.append(torch.tensor(0))
                    else:
                        pose_valid_list.append(torch.tensor(1))
                else:
                    pose = np.zeros(3)
                    pose_valid_list.append(torch.tensor(0))

                if self.lmk == 'insight':
                    lmk_path = os.path.join(imagebasepath, str(image_name)+'_lmk_insight.npy')
                else:
                    lmk_path = os.path.join(imagebasepath, str(image_name)+'_lmk.npy')
                if os.path.exists(lmk_path):
                    lmk = np.load(lmk_path, allow_pickle=True)
                    lmk_valid_list.append(torch.tensor(1))
                else:
                    lmk = np.zeros((68,2))
                    lmk_valid_list.append(torch.tensor(0))

                images_list.append(image)
                imagesfarl_list.append(imagefarl)
                arcface_list.append(torch.tensor(arcface_image))
                clip_list.append(clip_image)
                dinov2_list.append(dinov2_image)
                pose_list.append(torch.tensor(pose))
                lmk_list.append(torch.tensor(lmk))

        images_array = torch.from_numpy(np.array(images_list)).float()
        imagesfarl_array = torch.stack(imagesfarl_list).float()
        arcface_array = torch.stack(arcface_list).float()
        clip_array = torch.stack(clip_list).float()
        dinov2_array = torch.stack(dinov2_list).float()
        pose_array = torch.stack(pose_list).float()
        pose_valid_array = torch.stack(pose_valid_list)
        lmk_array = torch.stack(lmk_list).float()
        lmk_valid_array = torch.stack(lmk_valid_list)
        if len(currpredmesh_list) > 0:
            currpredmesh_array = torch.stack(currpredmesh_list) 
            bestallpredmesh_array = torch.stack(self.bestallpredmesh)
            actorpredmesh_array = torch.stack(self.actorpredmesh)
            actorpredflame_array = torch.stack(self.actorpredflame)
            gtmesh_array = torch.stack(self.gtmesh)
            gtflame_array = torch.stack(self.gtflame)
        else:
            currpredmesh_array = torch.tensor([]) 
            bestallpredmesh_array = torch.tensor([])
            actorpredmesh_array = torch.tensor([])
            actorpredflame_array = torch.tensor([])
            gtmesh_array = torch.tensor([])
            gtflame_array = torch.tensor([])

        return {
            'batchsize': torch.tensor(images_array.shape[0]),
            'image': images_array,
            'farl': imagesfarl_array,
            'arcface': arcface_array,
            'clip': clip_array,
            'dinov2': dinov2_array,
            'pose': pose_array,
            'pose_valid': pose_valid_array,
            'lmk': lmk_array,
            'lmk_valid': lmk_valid_array,
            'imagename': actor,
            'dataset': self.name,
            'flame': flame,
            'exp': exp,
            'currpredmesh': currpredmesh_array,
            'bestallpredmesh': bestallpredmesh_array,
            'actorpredmesh': actorpredmesh_array,
            'actorpredflame': actorpredflame_array,
            'gtmesh': gtmesh_array,
            'gtflame': gtflame_array
        }
