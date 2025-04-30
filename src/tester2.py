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
import torch.nn.functional as F
import torch.distributed as dist
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
#from pytorch3d.io import save_ply
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

from src.configs.config import cfg
from src.utils import util
from src.models.flame import FLAME
import trimesh
import scipy.io



from src.models.deca import DECA
from src.configs.deca_config import cfg as deca_cfg
from src.utils.utils import batch_orth_proj

sys.path.append("./src")
input_mean = 127.5
input_std = 127.5

#NOW_SCANS = '/home/wzielonka/datasets/NoWDataset/final_release_version/scans/'
#nowimages = '/home/wzielonka/datasets/NoWDataset/final_release_version/iphone_pictures/'
#NOW_BBOX = '/home/wzielonka/datasets/NoWDataset/final_release_version/detected_face/'
#STIRLING_PICTURES = '/home/wzielonka/datasets/Stirling/images/'
NOW_VALIDATION = '/work/pselvaraju_umass_edu/Project_FaceDiffusion/FACEDATA/NOW/imagepathsvalidation.txt'
AFLW2000_VALIDATION = '/work/pselvaraju_umass_edu/Project_FaceDiffusion/FACEDATA/AFLW2000/imagespathvalidation.txt'
AFLW2000MASK_VALIDATION = '/work/pselvaraju_umass_edu/Project_FaceDiffusion/FACEDATA/aflw2000_mask/imagespathvalidation.txt'
AFLW2000NOMASK_VALIDATION = '/work/pselvaraju_umass_edu/Project_FaceDiffusion/FACEDATA/aflw2000_nomask/imagespathvalidation.txt'
AFFECTNET_VALIDATION = '/work/pselvaraju_umass_edu/Project_FaceDiffusion/FACEDATA/AFFECTNET/imagespathvalidation.txt'
NOW_TEST = '/work/pselvaraju_umass_edu/Project_FaceDiffusion/FACEDATA/NOW/imagepathstest.txt'


class Tester2(object):
    #def __init__(self, model1, model2, model3, config=None, device=None, args=None):
    def __init__(self, models, config=None, cfgs=None, device=None, args=None, rankmodel=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.cfgs = cfgs

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
        self.render_mesh = True
        self.embeddings = {}
        self.nowimages = self.cfg.test.now_images
        self.affectnetimages = self.cfg.test.affectnet_images
        self.aflw2000images = self.cfg.test.aflw2000_images
        self.args = args
        self.rankmodel = rankmodel

        import clip
        self.pretrainedfarlmodel = self.cfg.model.pretrained
        self.farlmodel, self.farl_preprocess = clip.load("ViT-B/16", device="cpu")
        farl_state = torch.load(os.path.join(self.pretrainedfarlmodel, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth"))
        self.farlmodel.load_state_dict(farl_state["state_dict"], strict=False)
        self.farlmodel = self.farlmodel.to(self.device)

        # deca model
        self.model = {i:None for i in range(len(models))}
        for i in range(len(models)):
            self.model[i] = models[i].to(self.device)
            self.model[i].testing = True
            self.model[i].eval()

        flameModel = FLAME(self.cfg.model).to(self.device)
        self.faces = flameModel.faces_tensor.cpu()

        #deca_cfg.model.use_tex = False
        #deca_cfg.model.extract_tex = False
        #self.deca = DECA(config = deca_cfg, device='cuda')
        #self.faces = model_model.flameGenerativeModel.generator.faces_tensor.cpu()

        logger.info(f'[INFO]            {torch.cuda.get_device_name(device)}')

    def load_checkpoint(self, model, ckpt_path):
        #dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        checkpoint = torch.load(ckpt_path, map_location)
        #print(self.model.net)

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

    def process_image(self, img, app):
        images = []
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
        if bboxes.shape[0] != 1:
            logger.error('Face not detected!')
            return images
        i = 0
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        aimg = face_align.norm_crop(img, landmark=face.kps)
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)

        images.append(torch.tensor(blob[0])[None])

        return images

    def process_folder(self, folder, app):
        images = []
        image_names = []
        arcface = []
        count = 0
        files_actor = sorted(sorted(os.listdir(folder)))
        for file in files_actor:
            if file.startswith('._'):
                continue
            image_path = folder + '/' + file
            logger.info(image_path)
            image_names.append(image_path)
            count += 1

            ### NOW CROPPING
            scale = 1.6
            # scale = np.random.rand() * (1.8 - 1.1) + 1.1
            bbx_path = image_path.replace('.jpg', '.npy').replace('iphone_pictures', 'detected_face')
            bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
            left = bbx_data['left']
            right = bbx_data['right']
            top = bbx_data['top']
            bottom = bbx_data['bottom']

            image = imread(image_path)[:, :, :3]

            h, w, _ = image.shape
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * scale)

            crop_size = 224
            # crop image
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
            DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = image / 255.
            dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))

            arcface += self.process_image(cv2.cvtColor(dst_image.astype(np.float32) * 255.0, cv2.COLOR_RGB2BGR), app)

            dst_image = dst_image.transpose(2, 0, 1)
            images.append(torch.tensor(dst_image)[None])

        images = torch.cat(images, dim=0).float()
        arcface = torch.cat(arcface, dim=0).float()
        print("images = ", count)

        return images, arcface, image_names

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


    def test_now(self, name='now_test', id='nowcache', numface=1, istest='val', isocc=0):
        self.now(name, id, numface, istest, isocc)

    def test_aflw2000(self, name='aflw200_test', id='nocache', numface=2):
        self.aflw2000(name, id, numface)

    def test_realocc(self, name='realocc_test', id='nocache', numface=2, inputdir='.'):
        self.realocc(best_id=name, numface=int(numface), inputdir=inputdir)

    def test_affectnet(self, name='affectnet_test', id='nocache', numface=1):
        self.affectnet(name, id, numface)

    def test_youtube(self, name='youtube_test', id='nocache', numface=1, inputdir='.'):
        self.youtube(best_id=name, numface=int(numface), inputdir=inputdir)

    def test_stirling(self, best_model, id=None):
        self.load_checkpoint(best_model)
        name = self.get_name(best_model, id)
        self.stirling(name)

    def save_mesh(self, file, vertices):
        scaled = vertices * 1000.0
        save_ply(file, scaled.cpu(), self.model.render.faces[0].cpu())

        # mask = self.model.masking.get_triangle_whole_mask()
        # v, f = self.model.masking.get_masked_mesh(vertices, mask)
        # save_obj(file, v[0], f[0])

    def cache_to_cuda(self, cache):
        for key in cache.keys():
            i, a = cache[key]
            cache[key] = (i.to(self.device), a.to(self.device))
        return cache

    def create_now_cache(self):
        cache_path = os.path.join(self.cfg.test.cache_path, 'test_now_cache.pt')
        if os.path.exists(cache_path):
            cache = self.cache_to_cuda(torch.load(cache_path))
            return cache
        else:
            cache = {}

        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.4)

        for actor in tqdm(sorted(os.listdir(self.nowimages))):
            image_paths = sorted(glob(os.path.join(self.nowimages , actor , '*')))
            #print(image_paths, flush=True)
            for folder in image_paths:
                #print(folder)
                images, arcface, image_names = self.process_folder(folder, app)
                for i in range(len(image_names)):
                    cache[image_names[i]] = (images[i], arcface[i])
                #cache[folder] = (images, arcface)
                #print(cache.keys())
                #exit()

        torch.save(cache, cache_path) 
        return self.cache_to_cuda(cache)

    def create_stirling_cache(self):
        if os.path.exists('test_stirling_cache.pt'):
            cache = torch.load('test_stirling_cache.pt')
            return cache
        else:
            cache = {}

        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.1)

        cache['HQ'] = {}
        cache['LQ'] = {}

        for folder in ['Real_images__Subset_2D_FG2018']:
            for quality in ['HQ', 'LQ']:
                for path in tqdm(sorted(glob(STIRLING_PICTURES + folder + '/' + quality + '/*.jpg'))):
                    actor = path.split('/')[-1][:9].upper()
                    image = imread(path)[:, :, :3]
                    blobs = self.process_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), app)
                    if len(blobs) == 0:
                        continue
                    image = image / 255.
                    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
                    image = torch.tensor(image).cuda()[None]

                    if actor not in cache[quality]:
                        cache[quality][actor] = []
                    cache[quality][actor].append((image, blobs[0]))

        for q in cache.keys():
            for a in cache[q].keys():
                images, arcface = list(zip(*cache[q][a]))
                images = torch.cat(images, dim=0).float()
                arcface = torch.cat(arcface, dim=0).float()
                cache[q][a] = (images, arcface)

        torch.save(cache, 'test_stirling_cache.pt')
        return self.cache_to_cuda(cache)

    def update_embeddings(self, actor, arcface):
        if actor not in self.embeddings:
            self.embeddings[actor] = []
        self.embeddings[actor] += [arcface[i].data.cpu().numpy() for i in range(arcface.shape[0])]

    def stirling(self, best_id):
        logger.info(f"[TESTER] Stirling testing has begun!")
        self.model.eval()
        cache = self.create_stirling_cache()
        for quality in cache.keys():
            images_processed = 0
            self.embeddings = {}
            for actor in tqdm(cache[quality].keys()):
                images, arcface = cache[quality][actor]
                with torch.no_grad():
                    codedict = self.model.encode(images.cuda(), arcface.cuda())
                    opdict = self.model.decode(codedict, 0)

                self.update_embeddings(actor, codedict['arcface'])

                dst_actor = actor[:5]
                os.makedirs(os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality), exist_ok=True)
                dst_folder = os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor)
                os.makedirs(dst_folder, exist_ok=True)

                meshes = opdict['pred_canonical_shape_vertices']
                lmk = self.model.flame.compute_landmarks(meshes)

                for m in range(meshes.shape[0]):
                    v = torch.reshape(meshes[m], (-1, 3))
                    savepath = os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor, f'{actor}.ply')
                    self.save_mesh(savepath, v)
                    landmark_51 = lmk[m, 17:]
                    landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]
                    landmark_7 = landmark_7.cpu().numpy() * 1000.0
                    np.save(os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor, f'{actor}.npy'), landmark_7)
                    images_processed += 1

                    pred = self.model.render.render_mesh(meshes)
                    dict = {
                        'pred': pred,
                        'images': images
                    }

                    savepath = os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor, f'{actor}.jpg')
                    util.visualize_grid(dict, savepath, size=512)

            logger.info(f"[TESTER] Stirling dataset {quality} with {images_processed} processed!")

            # util.save_embedding_projection(self.embeddings, f'{self.cfg.output_dir}/stirling_test_{best_id}/stirling_{quality}_arcface_embeds.pdf')
    def aflw2000(self, best_id, id, numface=1, istest='val'):
        logger.info(f"[TESTER] AFLW2000 validation has begun!")
        for i in range(len(self.model.keys())):
            self.model[i].eval()
        #self.args.filename = self.args.filename.strip()
        #self.args.filename = '/project/pi_elearned_umass_edu/pselvaraju/pselvaraju/Project_FaceDiffusion/FACEDATA/COMA/coma_midframes/validation_occ_26_C/501.txt'
        valread = open(self.args.filename,'r')
#        for line in valread:
#            image_name = line.strip()
#            images = os.path.join(self.aflw2000images, line)
#        valread.close()
#        if not os.path.exists(images):
#            print("path does not exists")
#            print(images)
#            return

#        valread = open(AFLW2000_VALIDATION, 'r')
        allimages = []
        image_names = []
        for line in valread:
            line = line.strip()
            image_names.append(line)
            #allimages.append(os.path.join(self.aflw2000images, line))
            allimages.append(os.path.join(self.args.imagepath, line))
        valread.close()
        print(allimages)
        
        from pytorch_lightning import seed_everything
        r = np.random.randint(1000)
        seed_everything(r)
        for i in range(len(allimages)):
            images = allimages[i]
            image_name = image_names[i]
            image_name_noext = images[:-4]
            npyfile = (re.sub('jpg','npy', images))
            if not os.path.exists(npyfile):
                #print(npyfile, flush=True)
                continue
            imagefarl = self.farl_preprocess(PILImage.open(images))
            arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to('cuda')
            matfile = re.sub('jpg', 'mat', re.sub('arcface_input', 'FLAME_parameters', images))
            img450 = imread(re.sub('arcface_input/images', 'images', images))
            #kpt = scipy.io.loadmat(matfile)['pt3d_68'].T
            kpt = None # torch.Tensor(np.load(images[:-4]+'_aflwkpt.npy'))
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
                    'actor': '',
                    'type':'',
                    'kpt': kpt,
                    'img450': img450,
                    'outfile': ''}
            #self.decode(origimage, arcface, image_name, image_name1, best_id, id, numface, outfile='aflw2000')
            self.decode(result)

    def affectnet(self, best_id, id, numface=1, istest='val'):
        logger.info(f"[TESTER] Affectnet validation has begun!")
        for i in range(len(self.model.keys())):
            self.model[i].eval()

        if istest == 'val':
            valread = open(AFFECTNET_VALIDATION, 'r')
        else:
            valread = open(AFFECTNET_TEST, 'r')

        for line in valread:
            line = line.strip()
            print(line, flush=True)
            data = line.strip().split('/')
            image_name = data[0]
            images = os.path.join(self.affectnetimages, line)

            image_name1 = images[:-4]
            if id == 'affectnetcache':
                cache = self.create_affectnet_cache()
                img_cache = re.sub('arcface_input', images)
                origimage, arcface = cache[img_cache]
            else:
                arcfacepath = re.sub('jpg','npy', images)
                if not os.path.exists(arcfacepath):
                    print("arcface path does not exists", flush=True)
                    print(arcfacepath, flush=True)
                    continue
                arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to('cuda')
                origimage = imread(images)
                origimage1 = origimage / 255.
                origimage1 = origimage1.transpose(2, 0, 1)

            result = {'origimage': origimage,
                    'origimage1': origimage1,
                    'arcface': arcface,
                    'imgname': image_name,
                    'imgname1': image_name1,
                    'best_id': best_id,
                    'id': id,
                    'numface': numface,
                    'actor': '',
                    'type':'',
                    'kpt': None,
                    'img450': '',
                    'outfile': 'affectnet'}
            self.decode(result)
        valread.close()

    def realocc(self, best_id, numface=1, istest='val', inputdir='.'):
        logger.info(f"[TESTER] RealOcc validation has begun!")
        for i in range(len(self.model.keys())):
            self.model[i].eval()

        if os.path.isdir(inputdir):
            allimages = sorted(glob(inputdir+'/*.jpg'))
        elif os.path.isfile(inputdir):
            filename = open(inputdir,'r').read()
            allimages = [os.path.join(self.cfg.input_dir, filename)]
        #print(allimages)
        #exit()
        for line in allimages:
            line = line.strip()
            if (line.startswith('.') or ('lmk' in line) or ('aimg' in line)):
                continue
            data = line.strip().split('/')
            actor, image_name = data[-2], data[-1]
            images = os.path.join(inputdir, line)
            print("image = ", images, flush=True)
            imagefarl = self.farl_preprocess(PILImage.open(images))

            image_name_noext = images[:-4]
            arcfacepath = re.sub('jpg','npy', images)
            if not os.path.exists(arcfacepath):
                continue
            arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to('cuda')
            origimage = imread(images)
            normimage = origimage / 255.
            normtransimage = normimage.transpose(2, 0, 1)

            result = {'origimage': origimage,
                    'normimage': normimage,
                    'normtransimage': normtransimage,
                    'arcface': arcface,
                    'imgname': image_name,
                    'imagefarl': imagefarl,
                    'imgname_noext': image_name_noext,
                    'best_id': best_id,
                    'type': 'mask',
                    'id': id,
                    'numface': numface,
                    'actor': actor,
                    'kpt': None,
                    'img450': '',
                    'outfile': 'realocc'}
            self.decode(result)

    def youtube(self, best_id, numface=1, istest='val', inputdir='.'):
        logger.info(f"[TESTER] Youtube validation has begun!")
        for i in range(len(self.model.keys())):
            self.model[i].eval()

        allimages = sorted(glob(inputdir+'/*.jpg'))
        uncutimagedir = re.sub('arcface_input', 'frame_images_DB', inputdir)
        uncutimagedir = re.sub('YOUTUBE', 'YouTubeFaces', uncutimagedir)
        for line in allimages:
            line = line.strip()
            if (line.startswith('.') or ('lmk' in line) or ('aimg' in line)):
                continue
            data = line.strip().split('/')
            actor, vidnum, image_name = data[-3], data[-2], data[-1]
            images = os.path.join(inputdir, line)
            print("image = ", images, flush=True)
            uncutimage = os.path.join(uncutimagedir, image_name)

            image_name1 = images[:-4]
            arcfacepath = re.sub('jpg','npy', images)
            if not os.path.exists(arcfacepath):
                continue
            arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to('cuda')
            origimage = imread(images)
            origimage1 = origimage / 255.
            origimage1 = origimage1.transpose(2, 0, 1)

            uncutimage = imread(uncutimage)
            uncutimage = uncutimage / 255.
            uncutimage = uncutimage.transpose(2,0,1)


            result = {'origimage': origimage,
                    'origimage1': origimage1,
                    'uncutimage': uncutimage,
                    'arcface': arcface,
                    'imgname': image_name,
                    'imgname1': image_name1,
                    'best_id': best_id,
                    'id': id,
                    'numface': numface,
                    'actor': actor,
                    'type':vidnum,
                    'kpt': None,
                    'img450': '',
                    'outfile': 'youtube'}
            self.decode(result)

    def now(self, best_id, id, numface=1, istest='val', isocc=0):
        logger.info(f"[TESTER] NoW validation has begun!")
        for i in range(len(self.model.keys())):
            self.model[i].eval()

        if istest == 'val':
            valread = open(NOW_VALIDATION, 'r')
        else:
            valread = open(NOW_TEST, 'r')
        nowimages = self.nowimages
        if isocc:
            nowimages = re.sub('arcface_input', 'arcface_occluded_'+str(isocc)+'_input', nowimages)

        for line in valread:
            line = line.strip()
            print(line, flush=True)
            data = line.strip().split('/')
            actor, type, image_name = data[0], data[1], data[2]
            images = os.path.join(nowimages, line)
            imagefarl = self.farl_preprocess(PILImage.open(images))

            image_name_noext = images[:-4]
            if id == 'nowcache':
                cache = self.create_now_cache()
                img_cache = re.sub('arcface_input', 'final_release_version/iphone_pictures', images)
                origimage, arcface = cache[img_cache]
            else:
                arcfacepath = re.sub('jpg','npy', images)
                if not os.path.exists(arcfacepath):
                    continue
                arcface = torch.tensor(np.load(re.sub('jpg','npy', images))).float().to('cuda')
                origimage = imread(images)
                normimage = origimage / 255.
                normtransimage = normimage.transpose(2, 0, 1)
                #origimage = origimage / 255.
                #origimage = origimage.transpose(2, 0, 1)

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
                    'actor': actor,
                    'type':type,
                    'kpt': None,
                    'img450': '',
                    'outfile': 'now'}
            self.decode(result)
        valread.close()

    #def decode(self, origimage, arcface, image_name, image_name1, best_id, id, numface=1, outfile='now', actor='', type='', istest='val'):
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
        outfile= input['outfile']
        actor=input['actor']
        type=input['type']
        kpt=input['kpt']
        istest='val'

        interpolate = 224
        origimage_copy = origimage.copy()
        arcface_rank = arcface.clone()
        with torch.no_grad():
            
            #codedict1 = self.model[0].encode(torch.Tensor(normtransimage).unsqueeze(0).to('cuda'), arcface.unsqueeze(0))
            arcface1 = arcface.tile(100,1,1,1)
            img_tensor1 = torch.Tensor(normtransimage).tile(100,1,1,1).to('cuda')
            imgfarl_tensor1 = torch.Tensor(imagefarl).tile(100,1,1,1).to('cuda')
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
            #opdict1 = self.model[0].decode(codedict1, 0, withpose=False)

            #codedict4 = self.model[2].encode(torch.Tensor(normtransimage).unsqueeze(0).to('cuda'), arcface.unsqueeze(0))
            #transimage = origimage.transpose(2, 0, 1)
            #deca_codedict = self.deca.encode(torch.Tensor(transimage).unsqueeze(0).to('cuda'))
            #deca_codedict = self.deca.encode(torch.Tensor(decaimg).unsqueeze(0).to('cuda'))
            print("numface = ", numface)
            numface = 10
            if numface > 1:
                arcface = arcface.tile(numface,1,1,1)
                img_tensor = torch.Tensor(normtransimage).tile(numface,1,1,1).to('cuda')
                imgfarl_tensor = torch.Tensor(imagefarl).tile(numface,1,1,1).to('cuda')
                codedict2 = self.model[1].encode(img_tensor, arcface, imgfarl_tensor)
            else:
                codedict2 = self.model[1].encode(torch.Tensor(normtransimage).unsqueeze(0).to('cuda'), arcface.unsqueeze(0))

            opdict2 = []
            #rand = sortindex[0:3] # np.random.randint(0,100,3)
            #opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[rand[0]].tile(numface,1).float()))
            #opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[rand[1]].tile(numface,1).float()))
            #opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[rand[2]].tile(numface,1).float()))
            self.model[1].testing = True
            print(self.model[1].cfg.net.flame_dim, flush=True)
            print(numface, flush=True)
           
            from pytorch_lightning import seed_everything
            loops = int(100/numface)
            for i in range(loops):
                r = np.random.randint(1000)
                seed_everything(r)
                opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[maxindex].tile(numface,1).float()))
            #seed_everything(22)
            #opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[maxindex].tile(20,1).float()))
            #seed_everything(47)
            #opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[maxindex].tile(20,1).float()))
            #seed_everything(52)
            #opdict2.append(self.model[1].decode(codedict2, 0, shapecode=shape[maxindex].tile(20,1).float()))


            #opdict3 = None
            #if len(self.model) == 4:
            #    expression = opdict2['pred_flameparam']
            #    print("in opdict3", flush=True)
            #    opdict3 = self.model[3].decode(codedict4, 0, shapecode=shape.tile(numface,1).float(), expcode=expression.float(), rotcode=resnetrot.float())

            #opdict2 = self.model[1].testdecode(codedict2, 0, shapeparam=shape.tile(numface,1), rotparam=syn_pose.tile(numface,1))
            #opdict2 = self.model[1].testdecode(codedict2, 0, shapeparam=shape.tile(numface,1), rotparam=deca_pose.tile(numface,1))
            #opdict2 = self.model2.testdecode(codedict2, 0)

        #type = folder.split('/')[-1]
        os.makedirs(os.path.join(self.cfg.output_dir, f'{outfile}','flamesample'), exist_ok=True)
        flame_dst_folder = os.path.join(self.cfg.output_dir, f'{outfile}', 'flamesample', actor, type)
        os.makedirs(os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample'), exist_ok=True)
        shape_dst_folder = os.path.join(self.cfg.output_dir, f'{outfile}', 'shapesample', actor, type)
        #print("flame_dst_folder = ", flame_dst_folder, flush=True)
        os.makedirs(flame_dst_folder, exist_ok=True)
        os.makedirs(shape_dst_folder, exist_ok=True)
        print(flame_dst_folder, flush=True)

        image_name = re.sub('arcface_input/','',image_name)
        a = image_name
        savepath = os.path.split(os.path.join(flame_dst_folder, a))[0]
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.split(os.path.join(shape_dst_folder, a))[0]
        os.makedirs(savepath, exist_ok=True)
        #for ni in range(3):

        #front_face_vertices = np.load()
        pred_front = []
        all_pred_flame_meshes = []
        for ni in range(loops):
            if 'pred_mesh' in opdict2[ni]:
                pred_flame_meshes = opdict2[ni]['pred_mesh']
                ##### Get the shapes with maximum variations ############
                #distances = torch.cdist(front_face, front_face)
                findices = np.load('/work/pselvaraju_umass_edu/Project_FaceDiffusion/DIFFFACE/pretrained/frontindices.npy')
                print(pred_flame_meshes.shape)
                pred_front.append(pred_flame_meshes[:,findices].reshape(numface,-1))
                all_pred_flame_meshes.append(pred_flame_meshes)
            else:
                continue
        pred_front = torch.vstack(pred_front)
        print(pred_front.shape)
        all_pred_flame_meshes = torch.vstack(all_pred_flame_meshes)

#        for ni in range(1):
#            if 'pred_mesh' in opdict2[ni]:
#                pred_flame_meshes = opdict2[ni]['pred_mesh']
#                ##### Get the shapes with maximum variations ############
#                #distances = torch.cdist(front_face, front_face)
#                findices = np.load('/work/pselvaraju_umass_edu/Project_FaceDiffusion/DIFFFACE/pretrained/frontindices.npy')
#                pred_front = pred_flame_meshes[:,findices].reshape(numface,-1)
#                mean_front = torch.mean(pred_front, axis=0)
#                residual = pred_front - mean_front
#                std = torch.std(pred_front,axis=1)
#                #print(std)
#                distances = torch.cdist(mean_front.unsqueeze(0), residual)
#                expsortindex = torch.argsort(distances, descending=True)[0]
#                #print(distances)
#                #print(expsortindex)
#                #expsortindex = torch.argsort(std, descending=True)
#            else:
#                continue

            #selectnum = np.random.randint(0, numface, 20)
            #for num in range(20):
            #for num in range(numface):
        #mean_front = torch.mean(pred_front, axis=0)
        #residual = pred_front - mean_front
        #std = torch.std(pred_front,axis=1)
        #print(std)
        #distances = torch.cdist(mean_front.unsqueeze(0), residual)
        #expsortindex = torch.argsort(distances, descending=True)[0]
        #expsortindex = torch.argsort(std, descending=True)
        expsortindex = np.arange(15)

        for num in range(15):
            currname = a[:-4]+'_'+str(0)+'_'+str(num)+'.jpg'
            #print("currname = ", currname, flush=True)
            savepath = os.path.join(flame_dst_folder, currname.replace('jpg', 'ply'))
            print(savepath, flush=True)
            #trimesh.Trimesh(vertices=pred_flame_meshes[selectnum[num]].cpu() * 1000.0, faces=self.faces, process=False).export(savepath)
            #trimesh.Trimesh(vertices=pred_flame_meshes[num].cpu() * 1000.0, faces=self.faces, process=False).export(savepath)
            trimesh.Trimesh(vertices=all_pred_flame_meshes[expsortindex[num]].cpu() * 1000.0, faces=self.faces, process=False).export(savepath)

        #for num in range(3):
        for num in range(1):
            currname = a[:-4]+'_'+str(num)+'.jpg'
            saveshapepath = os.path.join(shape_dst_folder, currname.replace('jpg', 'ply'))
            #trimesh.Trimesh(vertices=pred_shape_meshes[rand[num]].cpu() * 1000.0, faces=self.faces, process=False).export(saveshapepath)
            trimesh.Trimesh(vertices=pred_shape_meshes[maxindex[num]].cpu() * 1000.0, faces=self.faces, process=False).export(saveshapepath)

