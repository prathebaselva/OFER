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
import sys
from glob import glob

import cv2
from PIL import Image as PILImage
import numpy as np
import torch
import re
import torch.distributed as dist
import torch.nn.functional as F
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


sys.path.append("./src")
input_mean = 127.5
input_std = 127.5


class Tester(object):
    def __init__(self, rankmodel, config=None, device=None, args=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
        self.render_mesh = True
        self.args = args

        # deca model
        self.model = rankmodel.to(self.device)
        self.model.testing = True
        self.model.eval()

        flameModel = FLAME(self.cfg.model).to(self.device)
        self.faces = flameModel.faces_tensor.cpu()

        logger.info(f'[INFO]            {torch.cuda.get_device_name(device)}')


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


    def test_realocc(self, name='now_test',numface=1, filename='', predfolder=''):
        self.realocc(name, numface, filename, predfolder)


    def save_mesh(self, file, vertices):
        scaled = vertices * 1000.0
        save_ply(file, scaled.cpu(), self.model.render.faces[0].cpu())


    def realocc(self,name,  numface, filename='', predfolder=''):
        self.model.eval()
        print("filename = ", filename)
        if not os.path.exists(filename):
            return 
        valread = open(filename,'r')
        allimages = []
        image_names = []
        predmeshes = []

        for line in valread:
            image_name = line.strip()
            image_names.append(image_name)
            images = os.path.join(self.cfg.input_dir, image_name)
            allimages.append(images)
            image_name = re.sub('arcface_input/','',image_name)
            predmeshes.append(os.path.join(predfolder, image_name[:-4]))
        valread.close()
        self.getrealoccsamples(allimages, numface, image_names, predmeshes)


    def getrealoccsamples(self, allimages, numface, image_names, predmeshes):
        import clip
        self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
        farl_state = torch.load(os.path.join(self.cfg.model.pretrained, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth" ))
        self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
        self.farlmodel = self.farlmodel.to(self.device)
        count = 1
        for i in range(len(image_names)):
            count += 1
            images = allimages[i]
            image_name = image_names[i]
            predmesh = predmeshes[i]
            allpredmesh = []
            allnpyfile = []
            imagefarl = self.farl_preprocess(PILImage.open(images))

            arcpath = re.sub('jpg','npy', images)
            arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to('cuda')
            origimage = imread(images)
            normimage = origimage / 255.
            normtransimage = origimage.transpose(2, 0, 1)

            with torch.no_grad():
                if numface > 1:
                    arcface = arcface.unsqueeze(0).to('cuda')
                    img_tensor = torch.Tensor(normtransimage).unsqueeze(0).to('cuda')
                    imgfarl_tensor = torch.Tensor(imagefarl).unsqueeze(0).to('cuda')
                    codedict, _ = self.model.encode(img_tensor, arcface, imgfarl_tensor)
                else:
                    codedict, _ = self.model.encode(torch.Tensor(origimage).unsqueeze(0).to('cuda'), arcface.unsqueeze(0))
                for j in range(numface):
                    mesh = trimesh.load(predmesh+'_'+str(j)+'.ply')
                    allpredmesh.append(torch.tensor(np.array(mesh.vertices)).float())
                allpredmesh = torch.stack(allpredmesh).to('cuda')
                opdict = self.model.decodetest(codedict, allpredmesh, epoch=0, withpose=False, numsamples=numface)
                per_vertex_error = opdict['pred_vertex_error']
                if 'pred_mesh_full' in opdict:
                    pred_flame_meshes = opdict['pred_mesh_full']
                if 'softmax_value' in opdict:
                    smax = opdict['softmax_value']

                print("pred vertex_error", per_vertex_error)
                #if 'score' in self.cfg.net.rankarch:
                minindex = torch.argmin(smax, dim=-1)
                maxindex = torch.argmax(smax, dim=-1)
                sindex = torch.argsort(smax, dim=-1)
                sindex = sindex.squeeze()
                print("sindex = ", sindex, flush=True)
                print(smax, flush=True)
                print("smax = ", smax, flush=True)
                print("min index = ", minindex)
                print("min val = ", smax[0][minindex])
                print("max index = ",maxindex)
                print("max val = ", smax[0][maxindex], flush=True)

            image_name = re.sub('arcface_input/','',image_name)
            a = image_name
            p = os.path.join(self.cfg.output_dir, 'minflamesample', image_name)
            minpath, a = os.path.split(p)
            os.makedirs(p, exist_ok=True)
            p = os.path.join(self.cfg.output_dir, 'maxflamesample', image_name)
            maxpath, a = os.path.split(p)
            os.makedirs(p, exist_ok=True)
            p = os.path.join(self.cfg.output_dir, 'indexflamesample', image_name)
            indexpath, a = os.path.split(p)
            os.makedirs(p, exist_ok=True)


            for num in range(1):
                currname = a[:-4]+'_'+str(num)+'.jpg'
                minsavepath = os.path.join(minpath, currname.replace('jpg', 'ply'))
                maxsavepath = os.path.join(maxpath,  currname.replace('jpg', 'ply'))
                print(minsavepath, flush=True)

                trimesh.Trimesh(vertices=allpredmesh[maxindex].squeeze().cpu().numpy() , faces=self.faces, process=False).export(maxsavepath)
                trimesh.Trimesh(vertices=allpredmesh[minindex].squeeze().cpu().numpy() , faces=self.faces, process=False).export(minsavepath)

                np.save(os.path.join(indexpath, currname.replace('jpg', 'npy')), sindex.cpu().numpy())

    def getmaxsampleindex(self, arcface, normtransimage, imagefarl, allpredmesh, numface=100):
            arcface = arcface.unsqueeze(0).to('cuda')
            img_tensor = torch.Tensor(normtransimage).unsqueeze(0).to('cuda')
            imgfarl_tensor = torch.Tensor(imagefarl).unsqueeze(0).to('cuda')
            codedict, _ = self.model.encode(img_tensor, arcface, imgfarl_tensor)
            allpredmesh = 1000*(allpredmesh).to('cuda')
            opdict = self.model.decodetest(codedict, allpredmesh, epoch=0, withpose=False, numsamples=numface)# cam=deca_codedict['cam'])
            per_vertex_error = opdict['pred_vertex_error']
            if 'pred_mesh_full' in opdict:
                pred_flame_meshes = opdict['pred_mesh_full']
            if 'softmax_value' in opdict:
                smax = opdict['softmax_value']

            print("pred vertex_error", per_vertex_error)
            maxindex = torch.argmax(smax, dim=-1)
            sindex = torch.argsort(smax, dim=-1, descending=True)
            sindex = sindex.squeeze()
            return maxindex, sindex
