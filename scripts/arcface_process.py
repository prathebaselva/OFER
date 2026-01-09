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


import argparse
import os
import re
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
#import trimesh
from insightface.app import FaceAnalysis, MaskRenderer
from insightface.app.common import Face
from insightface.utils import face_align
import face_alignment
from loguru import logger
#from skimage.i
#from imageio import imread, imsave
#import imageio
import imageio
from scipy.io import loadmat
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
import math

import sys
sys.path.append('../')

from SynergyNet.synergy3DMM import SynergyNet
synmodel = SynergyNet()
facemodel = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

now_valpath = '../NOW/imagepathsvalidation.txt'
now_testpath = '../NOW/imagepathstest.txt'

#from utils import util
input_mean = 127.5
input_std = 127.5

def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False

def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))

def draw_bbox(img, face, landmarks, outfile):
    box = face.bbox.astype(np.int)
    color = (0,0,255)
    cv2.rectangle(img, (box[0], box[1], box[2], box[3]), color, 2)
    cv2.imwrite(outfile+'_bbox.jpg', img)

def draw_landmark(img, face, landmarks, outfile):
    color = (0,0,255)
    for i in range(len(landmarks)):
        cv2.circle(img, (int(landmarks[i][0]), int(landmarks[i][1])), 1, color, 2)
    cv2.imwrite(outfile+'_lmk_insight.jpg', img)

def get_center(bboxes, img):
    img_center = img.shape[0] // 2, img.shape[1] // 2
    size = bboxes.shape[0]
    distance = np.Inf
    j = 0
    for i in range(size):
        x1, y1, x2, y2 = bboxes[i, 0:4]
        dx = abs(x2 - x1) / 2.0
        dy = abs(y2 - y1) / 2.0
        current = dist((x1 + dx, y1 + dy), img_center)
        if current < distance:
            distance = current
            j = i

    return j

def get_arcface_input(face,img, filepath, image_size=112):
    #aimg = face_align.norm_crop(img, landmark=face.kps)
    #M = face_align.estimate_norm(face.kps)
    #aimg = cv2.warpAffine(img, M, (image_size, image_size),0.0)

    aimg = face_align.norm_crop(img, landmark=face.kps, image_size=image_size)
    #skimage.io.imsave(filepath+'_aimg_assuch1.jpg', aimg)
    cv2.imwrite(filepath + '_aimg_assuch.jpg', aimg)
    #blob = cv2.dnn.blobFromImages([aimg], 1.0/input_std, (image_size, image_size), (input_mean, input_mean, input_mean), swapRB=True)
    aimg_rgb = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
    #blob = cv2.dnn.blobFromImages([aimg_rgb], 1.0/input_std, (image_size, image_size), (input_mean, input_mean, input_mean), swapRB=False)
    blob = cv2.dnn.blobFromImages([aimg], 1.0/input_std, (image_size, image_size), (input_mean, input_mean, input_mean), swapRB=True)
    #aimg_rgb = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filepath + '_aimg.jpg', aimg_rgb)
    np.save(filepath, blob[0])
    return blob[0], aimg_rgb


def process1(img, app, path, name):
    filepath = str(Path(path, name))
    if not os.path.exists(filepath+'.jpg'):
        return 0

    dst_image = cv2.imread(filepath + '.jpg')

    pose = synmodel.get_pose_output(dst_image) 
    pose = np.array(pose).squeeze()
    if (pose is not None) and  (len(pose) > 0):
        np.save(filepath+'_pose.npy', pose)

    out = facemodel.get_landmarks_from_image(dst_image)
    dst_image2 = dst_image.copy()
    dst_image3 = dst_image.copy()
    if out is not None and len(out) > 0:
        out = out[0]
        np.save(filepath+'_lmk.npy', out)

    faces = app.get(dst_image3, max_num=1)
    if faces is not None and len(faces) > 0:
        landmarks = faces[0].landmark_3d_68[:,:2]
        np.save(filepath+'_lmk_insight.npy', landmarks)

def processwithposeandlmk(img, app, path, name, aflwkpt=None):
    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
    if bboxes.shape[0] == 0:
        print("no bbox found")
        print(name, flush=True)
        return 0
    i = get_center(bboxes, img)
    bbox = bboxes[i, 0:4]
    det_score = bboxes[i, 4]
    kps = None
    if kpss is not None:
        kps = kpss[i]

    filepath = str(Path(path, name))
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    blob, aimg = get_arcface_input(face, img, filepath)

    scale = 1.6
    #print(bboxes[i,0:4], flush=True)
    left, top, right, bottom = bboxes[i, 0:4]
    h, w, _ = img.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    crop_size = 224

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    img = img / 255.
    dst_image = warp(img, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image  = cv2.cvtColor(dst_image.astype(np.float32) * 255.0, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath + '.jpg', dst_image)

    dst_kpt = None
    if aflwkpt is not None:
        aflwkpt[:,2] = np.ones(68)
        dst_kpt = np.dot(tform, aflwkpt.T).T
        np.save(filepath+'_aflwkpt.npy', dst_kpt)

    pose = synmodel.get_pose_output(dst_image) 
    pose = np.array(pose).squeeze()
    if (pose is not None) and  (len(pose) > 0):
        np.save(filepath+'_pose.npy', pose)

    out = facemodel.get_landmarks_from_image(dst_image)
    dst_image2 = dst_image.copy()
    dst_image3 = dst_image.copy()
    dst_image4 = dst_image.copy()
    if out is not None and len(out) > 0:
        out = out[0]
        color = (0,0,255)
        for i in range(len(out)):
            cv2.circle(dst_image2, (int(out[i][0]), int(out[i][1])), 1, color, 2)
        cv2.imwrite(filepath+'_lmk.jpg', dst_image2)
        np.save(filepath+'_lmk.npy', out)

    if dst_kpt is not None:
        for i in range(len(dst_kpt)):
            cv2.circle(dst_image4, (int(dst_kpt[i][0]), int(dst_kpt[i][1])), 1, color, 2)
        cv2.imwrite(filepath+'_dst_kpt.jpg', dst_image4)

    faces = app.get(dst_image3, max_num=1)
    if faces is not None and len(faces) > 0:
        landmarks = faces[0].landmark_3d_68[:,:2]
        color = (0,0,255)
        for i in range(len(landmarks)):
            cv2.circle(dst_image3, (int(landmarks[i][0]), int(landmarks[i][1])), 1, color, 2)
        cv2.imwrite(filepath+'_lmk_insight.jpg', dst_image3)
        np.save(filepath+'_lmk_insight.npy', landmarks)


def process(img, app, path, name):
    try:
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
    except:
        print("issue with model detect")
        return False

    if bboxes.shape[0] == 0:
        print("no bbox found")
        print(name, flush=True)
        return 0
    i = get_center(bboxes, img)
    bbox = bboxes[i, 0:4]
    det_score = bboxes[i, 4]
    kps = None
    if kpss is not None:
        kps = kpss[i]

    filepath = str(Path(path, name))
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    blob, aimg = get_arcface_input(face, img, filepath)

    scale = 1.6
    #print(bboxes[i,0:4], flush=True)
    left, top, right, bottom = bboxes[i, 0:4]
    h, w, _ = img.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    crop_size = 224

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    img = img / 255.
    dst_image = warp(img, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image  = cv2.cvtColor(dst_image.astype(np.float32) * 255.0, cv2.COLOR_RGB2BGR)
    print(filepath, flush=True)
    cv2.imwrite(filepath + '.jpg', dst_image)
    return True

def process_occluded(img, app, path, name, img_occ):
    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
    if bboxes.shape[0] == 0:
        return 0
    i = get_center(bboxes, img)
    bbox = bboxes[i, 0:4]
    det_score = bboxes[i, 4]
    kps = None
    if kpss is not None:
        kps = kpss[i]
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    blob, aimg, blob_occ, aimg_occ = get_arcface_input(face, img, img_occ)
    file = str(Path(path, name))
    np.save(file, blob_occ)
    cv2.imwrite(file + '.jpg', face_align.norm_crop(img_occ, landmark=face.kps, image_size=224))
    return 1

def processinstance(args, app, image_size=224):
    dst = Path(args.o)
    dst.mkdir(parents=True, exist_ok=True)
    processes = []
    instance = args.r
    inputdir = args.i
    outputdir = args.o
    actor = args.a

   ################# LYHM #########################################
    count = 0
    if instance == 'LYHM':
        actor = str(actor).zfill(5)
        images = glob(os.path.join(inputdir,f'{actor}/*/*.png'))
        for im in images:
            data = im.split('/')
            image_name = data[-1]
            if image_name.startswith('.'):
                continue
            image_name = image_name.split('.')[0]
            subactor = data[-2]
            imread = cv2.imread(im)
            os.makedirs(os.path.join(outputdir, actor, subactor), exist_ok =True)
            path = os.path.join(outputdir, actor, subactor)
            if not process(imread, app, path, image_name):
                print("image not processed = ", im, flush=True)
                continue
            #if not processwithposeandlmk(im1, app, path, image_name):
            #if not process1(im1, app, path, image_name):
                #continue

    elif instance == 'FRGC':
        for actor in sorted(glob(inputdir+'/*')):
            images = glob(f'{actor}/*/*.jpg')
            for im in images:
                data = im.split('/')
                image_name = data[-1]
                if image_name.startswith('.'):
                    continue
                image_name = image_name[:-4]
                subactor = data[-2]
                imread = cv2.imread(im)
                os.makedirs(os.path.join(outputdir, actor, subactor), exist_ok =True)
                path = os.path.join(outputdir, actor, subactor)
                if os.path.exists(path, image_name+'.npy'):
                    continue
                if not process(imread, app, path, image_name):
                #if not process1(im1, app, path, image_name):
                    continue

    elif instance == 'FLORENCE':
        for actor in sorted(glob(inputdir+'/*')):
            images = glob(f'{actor}/*.jpg')
            actorname = Path(actor).stem
            for im in images:
                image_name = im.split('/')[-1]
                if image_name.startswith('.'):
                    continue
                image_name = image_name.split('.')[0]
                im1 = imread(im)
                os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
                path = os.path.join(outputdir, actorname)
                if not processwithposeandlmk(im1, app, path, image_name):
                #if not process1(im1, app, path, image_name):
                    continue

    elif (instance == 'STIRLING' or instance == 'STIRLING_HQ' or instance == 'STIRLING_LQ'):
        for im in sorted(glob(inputdir+'/*.jpg')):
            image_name = im.split('/')[-1]
            if image_name.startswith('.'):
                continue
            image_name = image_name.split('.')[0]
            im1 = imread(im)
            path = outputdir
            if not processwithposeandlmk(im1, app, path, image_name):
            #if not process1(im1, app, path, image_name):
                continue

    elif instance == 'FACEWAREHOUSE':
        for actor in sorted(glob(inputdir+'/*')):
            images = glob(f'{actor}/TrainingPose/*.png')
            actorname = Path(actor).stem
            #print(actorname, flush=True)
            for im in images:
                image_name = im.split('/')[-1]
                if image_name.startswith('.'):
                    continue
                image_name = image_name.split('.')[0]
                im1 = imread(im)
                os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
                path = os.path.join(outputdir, actorname) 
                if not processwithposeandlmk(im1, app, path, image_name):
                    continue
            #final_im.save(os.path.join(outputdir, actorname, 'TrainingPose', image_name))
    elif instance == 'COMA':
        for images in sorted(glob(inputdir+'/*/*/*.jpg')):
            print(images)
            data = images.split('/')
            actorname = data[-3]
            expression = data[-2]
            image = data[-1].split('.')
            index, profile = image[1], image[2]
            ply = os.path.join(inputdir, actorname, expression, expression+'.'+index+'.ply')
            if not os.path.exists(ply):
                continue
            im = imread(images)
            image_name = expression+'.'+index+'.'+profile
            os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
            path = os.path.join(outputdir, actorname)
            if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'COMA1':
        from skimage.io import imread
        for images in sorted(glob(inputdir+'/*.jpg')):
            #print(images)
            data = images.split('/')
            actorname = data[-2]
            image_name = data[-1][:-4]
            if image_name.startswith('.'):
                continue
            im = imread(images)
            os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
            path = os.path.join(outputdir, actorname)
            if os.path.exists(path, image_name+'.npy'):
                continue
            if not process(im, app, path, image_name):
                print(image_name)
                print("not processed", flush=True)
                continue

    elif instance == 'COMAOCC':
        from skimage.io import imread
        for images in sorted(glob(inputdir+'/*/*.jpg')):
            #print(images)
            data = images.split('/')
            actorname = data[-2]
            image_name = data[-1][:-4]
            if image_name.startswith('.'):
                continue
            im = imread(images)
            os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
            path = os.path.join(outputdir, actorname)
            #if os.path.exists(path, image_name+'.npy'):
            #    continue
            if not process(im, app, path, image_name):
                print(image_name)
                print("not processed", flush=True)
                continue

    elif instance == 'TEMPEH':
        for images in sorted(glob(inputdir+'/*/*/*.png')):
            data = images.split('/')
            actorname = data[-4]
            expression = data[-3]
            index = data[-2]
            image_name = data[-1]
            if image_name.startswith('.') or os.path.getsize(images) <=0:
                continue
            image = image_name.split('.')
            index, profile = image[1], image[2]
            image_name = expression+'.'+index+'.'+profile
            
            try:
                im = imread(images)
                os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
                path = os.path.join(outputdir, actorname)
                filepath = os.path.join(path, image_name+'.jpg')
                if os.path.exists(filepath):
                    continue
                if not processwithposeandlmk(im, app, path, image_name):
                    continue
            except:
                print(images, flush=True)

    elif instance == 'IMAVATAR':
        for images in sorted(glob(inputdir+'/*.png')):
            image_name = images.split('/')[-1].split('.')[0]
            im = imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'AFFECTNET':
        for images in sorted(glob(inputdir+'/*.png')):
            image_name = images.split('/')[-1].split('.')[0]
            im = imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'FFHQ':
        for images in sorted(glob(inputdir+'/*.png')):
            image_name = images.split('/')[-1].split('.')[0]
            im = imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'CELEBA':
        for images in sorted(glob(inputdir+'/*.jpg')):
            image_name = images.split('/')[-1].split('.')[0]
            im = imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'HRN':
        for images in sorted(glob(inputdir+'/*.png')):
            image_name = images.split('/')[-1].split('.')[0]
            im = imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'AFLW2000':
        for images in sorted(glob(inputdir+'/*.jpg')):
            image_name = images.split('/')[-1].split('.')[0]
            print(image_name)
            im = imread(images)
            matfile = re.sub('images', 'FLAME_parameters', re.sub('jpg','mat', images))
            kpt = loadmat(matfile)['pt3d_68'].T
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            #if not processwithposeandlmk(im, app, path, image_name, aflwkpt=kpt):
            if not process(im, app, path, image_name, aflwkpt=kpt):
                continue

    elif instance == 'AFLW2000MASK':
        print(instance, flush=True)
        for images in sorted(glob(inputdir+'/*.jpg')):
            image_name = images.split('/')[-1].split('.')[0]
            print(image_name, flush=True)
            im = imageio.imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            print("outputdir = ", outputdir)
            #exit()
            path = os.path.join(outputdir)
            #if not processwithposeandlmk(im, app, path, image_name, aflwkpt=kpt):
            if not process(im, app, path, image_name):
                continue

    elif instance == 'FACESCAPE':
        for images in sorted(glob(inputdir+"/*/*.jpg")):
            #print(name)
            data = images.split('/')
            imgname = data[-1].split('.')[0]
            image_name = data[-3]+'_'+data[-2]+'_'+imgname
            im = imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not process(im, app, path, image_name):
            #if not processwithposeandlmk(im, app, path, image_name):
                continue


    elif instance == 'FACESCAPETEST':
        print(inputdir, flush=True)
        for images in sorted(glob(inputdir+"/*.png")):
            #print(name)
            data = images.split('/')
            imgname = data[-1].split('.')[0]
            image_name = imgname # data[-3]+'_'+data[-2]+'_'+imgname
            im = imread(images)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not process(im, app, path, image_name):
            #if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'NOW':
        valpath = open(now_valpath, 'r')
        #valpath = open(now_testpath, 'r')
        valimages = []
        for line in valpath:
            line = line.strip()
            valimages.append(line)
        valpath.close()

        for images in sorted(glob(inputdir+'/*/*/*.jpg')):
            images = images.strip()

            data = images.split('/')
            actor, exp, image = data[-3], data[-2], data[-1]
            valimg = os.path.join(actor, exp, image)
            if (image.startswith('.') or ('lmk' in image) or ('aimg' in image)):
                continue

            if not (valimg in valimages):
                print(valimg)
                continue

            image_name = image.split('.')[0]
            im = imread(images)
            os.makedirs(os.path.join(outputdir, actor, exp), exist_ok =True)
            path = os.path.join(outputdir, actor, exp)
            if not processwithposeandlmk(im, app, path, image_name):
                continue
            #break
            #if not process1(im, app, path, image_name):
            #    continue
    elif instance == 'YOUTUBE':
        for images in sorted(glob(inputdir+'/*.jpg')):
            images = images.strip()

            data = images.split('/')
            #actor, exp, image_name = data[-3], data[-2], data[-1].split('.')[0]
            actor, vidnum, image =  data[-3], data[-2], data[-1]
            if (image.startswith('.') or ('lmk' in image) or ('aimg' in image)):
                continue

            image_name = image[:-4]
            im = imread(images)
            os.makedirs(os.path.join(outputdir, actor, vidnum), exist_ok =True)
            path = os.path.join(outputdir, actor, vidnum)
            if not processwithposeandlmk(im, app, path, image_name):
                continue

    elif instance == 'REALOCCNOACTOR':
        from skimage.io import imread
        for im in sorted(glob(inputdir+'/*.jpg')):
                image_name = im.split('/')[-1]
                if image_name.startswith('.'):
                    continue
                image_name = image_name.split('.')[0]
                im1 = imread(im)
                im2 = cv2.imread(im)
                os.makedirs(os.path.join(outputdir), exist_ok =True)
                path = os.path.join(outputdir)
                if not process(im1, app, path, image_name):
                    continue

    elif instance == 'REALOCC':
        from skimage.io import imread
        for actor in sorted(glob(inputdir+'/*')):
            images = glob(f'{actor}/*.jpg')
            actorname = Path(actor).stem
            print(actorname, flush=True)
            for im in images:
                image_name = im.split('/')[-1]
                if image_name.startswith('.'):
                    continue
                image_name = image_name.split('.')[0]
                im1 = imread(im)
                os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
                path = os.path.join(outputdir, actorname)
                if not process(im1, app, path, image_name):
                    continue

    elif instance == 'DIVERSE3D':
        from skimage.io import imread
        for image in sorted(glob(inputdir+'/*/*/*.jpg')):
            data = image.split('/')
            actorname = data[-3]
            mask = data[-2]
            image_name = data[-1]
            if image_name.startswith('.'):
                continue
            if 'gt' in image_name:
                continue
            image_name = actorname+'_'+mask+'_'+image_name[:-4]
            im1 = imread(image)
            os.makedirs(os.path.join(outputdir), exist_ok =True)
            path = os.path.join(outputdir)
            if not process(im1, app, path, image_name):
                continue

def main( args):
    device = 'cuda:0'
    #Path(args.o).mkdir(exist_ok=True, parents=True)
    app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'landmark_3d_68'])
    app.prepare(ctx_id=0, det_size=(224, 224))

    logger.info(f'Processing has started...')
    paths = processinstance(args, app)
    #logger.info(f'Processing finished. Results has been saved in {args.o}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('-i', default='demo/input', type=str, help='Input folder with images')
    parser.add_argument('-r', default='FLORENCE', type=str, help='Input instance')
    parser.add_argument('-o', default='demo/arcface', type=str, help='Processed images for MICA input')
    parser.add_argument('-a', default='actor', type=str, help='Actor')

    args = parser.parse_args()
    main(args)
