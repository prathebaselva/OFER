import numpy as np
import os
import sys
from glob import glob


def facewarehouse(imgfolder, flamefolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    for imgpath in sorted(glob(imgfolder+'/*/*')):
        data = imgpath.split('/')
        img = data[-1]
        actor = data[-2]
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)

    for actor in npy.keys():
        print(actor)
        flamepath = os.path.join(flamefolder, actor)
        if os.path.exists(flamepath):
            fname = os.listdir(flamepath)
            npyfiles[actor].append(actor+'/'+fname[0])
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]
    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)


def facescape(imgfolder, dpmapfolder, outputfolder, datasetname):
    npy = {}
    npyfiles = {}
    for imgpath in sorted(glob(imgfolder+'/*')):
        img = imgpath.split('/')[-1]
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            actordata = (actorpath.split('.')[0]).split('_')
            actor = '_'.join([actordata[i] for i in range(len(actordata)-1)])
            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            print(actorpath)
            exit()
            npy[actor].append(actorpath)
    
    for actor in npy.keys():
        dpmappath = os.path.join(dpmapfolder, actor+'.png')
        if os.path.exists(dpmappath):
            npyfiles[actor].append(actor+'.png')
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def stirling(imgfolder, flamefolder, outputfolder, datasetname1):
# STIRLING
    data = datasetname1.split('_')
    datasetname, qual = data[0], data[1]
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    for fname in allf:
        print(fname)
        if (('.png' in fname) or ('.jpg' in fname)) and (not fname.startswith('.')) and (not 'lmk' in fname) and (not 'aimg' in fname):
            actor = (fname.split('.')[0]).split('_')[0]
            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(qual+'/'+fname)
    
    for actor in npy.keys():
        flamepath = os.path.join(flamefolder, actor)
        print(flamepath)
        print(actor)
        if os.path.exists(flamepath):
            fname = os.listdir(flamepath)
            npyfiles[actor].append(actor+'/'+fname[0])
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname1+'.npy'), npyfiles)

def florence(imgfolder, flamefolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    for imgpath in sorted(glob(imgfolder+'/*/*')):
        img = imgpath.split('/')[-1]
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            actor = (imgpath[len(imgfolder):]).split('/')[0]
            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)
    
    for actor in npy.keys():
        flamepath = os.path.join(flamefolder, actor)
        if os.path.exists(flamepath):
            fname = os.listdir(flamepath)
            npyfiles[actor].append(actor+'/'+fname[0])
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def lyhm(imgfolder, flamefolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    for imgpath in sorted(glob(imgfolder+'/*/*/*')):
        #print(imgpath)
        img = imgpath.split('/')[-1]
        #exit()
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            print(actorpath)
            actor = (imgpath[len(imgfolder):]).split('/')[0]
            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)
    
    for actor in npy.keys():
        flamepath = os.path.join(flamefolder, actor)
        #print(flamepath)
        if os.path.exists(flamepath):
            fname = os.listdir(flamepath)
            npyfiles[actor].append(actor+'/'+fname[0])
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def aflw2000(imgfolder, flamefolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    for imgpath in sorted(glob(imgfolder+'/*')):
        #print(imgpath)
        img = imgpath.split('/')[-1]
        imgname = img.split('.')[0]
        #exit()
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and ((not img.startswith('._')) and (not 'lmk' in img)):
            actorpath = (imgpath[len(imgfolder):])
            actor = imgname 
            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)
    #print(npy)
    
    for actor in npy.keys():
        flamepath = os.path.join(flamefolder, actor+'.mat')
        #print(flamepath)
        if os.path.exists(flamepath):
            npyfiles[actor].append(actor+'.mat')
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    print(npyfiles)
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def coma(imgfolder, flamefolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    count = 0
    for imgpath in sorted(glob(imgfolder+'/*/*')):
        #print(imgpath)
        img = imgpath.split('/')[-1]
        data = img.split('.')
        expression = data[0]
        index = data[1]
        #exit()
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            print(actorpath)
            actor = (imgpath[len(imgfolder):]).split('/')[0]
            actor = actor+'__'+expression+'__'+index
            #print(actor)
            #exit()

            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)

    
    for actor in npy.keys():
        data = actor.split('__')
        print(data)
        actorname = data[0]
        expression = data[1]
        index = data[2]
        #plypath = os.path.join(flamefolder, actorname, expression, expression+'.'+index+'.ply')
        npypath = os.path.join(flamefolder, actorname, expression, expression+'.'+index+'.npy')
        #if os.path.exists(plypath):
        if os.path.exists(npypath):
            npyfiles[actor].append(npypath)
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]
        #print(npyfiles[actor])
        #exit()

    print(len(npyfiles))
    #print(npyfiles)
    #exit()
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def tempeh(imgfolder, flamefolder, outputfolder, datasetname):
    allf = os.listdir(imgfolder)
    npy = {}
    npyfiles = {}
    count = 0
    for imgpath in sorted(glob(imgfolder+'/*/*')):
        #print(imgpath)
        img = imgpath.split('/')[-1]
        data = img.split('.')
        expression = data[0]
        index = data[1]
        #exit()
        if (('.png' in imgpath) or ('.jpg' in imgpath)) and (not img.startswith('.')) and (not 'lmk' in img) and (not 'aimg' in img):
            actorpath = (imgpath[len(imgfolder):])
            print(actorpath)
            actor = (imgpath[len(imgfolder):]).split('/')[0]
            actor = actor+'__'+expression+'__'+index

            if actor not in npy:
                npy[actor] = []
                npyfiles[actor] = []
            npy[actor].append(actorpath)
    
    for actor in npy.keys():
        data = actor.split('__')
        #print(data)
        actorname = data[0]
        expression = data[1]
        index = data[2]
        #plypath = os.path.join(flamefolder, actorname, expression, expression+'.'+index+'.ply')
        #npzpath = os.path.join(flamefolder, actorname, expression, expression+'.'+index+'.npz')
        npypath = os.path.join(flamefolder, actorname, expression, expression+'.'+index+'.npy')
        if os.path.exists(npypath):
            npyfiles[actor].append(npypath)
            npyfiles[actor].append(npy[actor])
        else:
            del npyfiles[actor]

    print(len(npyfiles))
    np.save(os.path.join(outputfolder, datasetname+'.npy'), npyfiles)

def get_image_paths(imgfolder, flamefolder, outputfolder, datasetname):
    if datasetname == 'STIRLING_HQ':
        stirling(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'STIRLING_LQ':
        stirling(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'FLORENCE':
        florence(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'FACEWAREHOUSE':
        facewarehouse(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'LYHM':
        lyhm(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'AFLW2000':
        aflw2000(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'COMA':
        coma(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'TEMPEH':
        tempeh(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'FACESCAPE':
        facescape(imgfolder, flamefolder, outputfolder, datasetname)
    if datasetname == 'FACESCAPE_NEUTRAL':
        facescape(imgfolder, flamefolder, outputfolder, datasetname)
        
if __name__ == '__main__':
    get_image_paths(*sys.argv[1:])
