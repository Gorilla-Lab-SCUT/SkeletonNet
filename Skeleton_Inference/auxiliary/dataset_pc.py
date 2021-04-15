from __future__ import print_function
import torch.utils.data as data
import os.path
import errno
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import random
import math
import os
import sys
import scipy.io as sio
import h5py
import cv2
from PIL import Image
from config import ROOT_IMG, ROOT_PC, ROOT_SPLIT, SYNSET_PATH, \
                    RENDER_DIRNAME, SKE_FILENAME, TRAIN_SPLIT_FILENAME, TEST_SPLIT_FILENAME
from utils import *

class ShapeNet(data.Dataset):
    def __init__(self, rootimg = ROOT_IMG, rootpc = ROOT_PC, class_choice = "chair",
                 npoints_skeleton=2500, npoints_line = 2500, npoints_square = 5000,
                 balanced = False, train = True, SVR=True, num_views=6, gen_view=False, idx=0, rotate=False, white_bg=False):
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints_skeleton = npoints_skeleton
        self.npoints_line = npoints_line
        self.npoints_square = npoints_square
        self.datapath = []
        self.catfile = os.path.join(SYNSET_PATH)
        self.cat = {}
        self.meta = {}
        self.balanced = balanced
        self.train = train
        self.SVR = SVR
        self.num_views = num_views
        self.gen_view = gen_view
        self.idx=idx
        self.rotate = rotate
        self.white_bg = white_bg
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)
        empty = []
        for item in self.cat:
            if train:
                fns = open(os.path.join(ROOT_SPLIT, self.cat[item]+'_'+TRAIN_SPLIT_FILENAME),'r').readlines()
                fns = [fn.strip() for fn in fns]
            else:
                fns = open(os.path.join(ROOT_SPLIT, self.cat[item]+'_'+TEST_SPLIT_FILENAME),'r').readlines()
                fns = [fn.strip() for fn in fns]

            dir_img = os.path.join(self.rootimg, self.cat[item])
            fns_img = sorted(os.listdir(dir_img))
            dir_ske = os.path.join(self.rootpc, self.cat[item])
            #fns_ske = sorted(os.listdir(dir_ske))
            #fns = [val for val in fns if val + '.npz' in fns_ske and val in fns_img]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns)/float(len(fns_img)), "%")

            if len(fns) !=0:
                self.meta[item] = []
            for modname in fns:
                cat_mod_img_dir = os.path.join(dir_img, modname, RENDER_DIRNAME)
                cat_mod_ske_dir = os.path.join(dir_ske, modname)
                self.meta[item].append((cat_mod_img_dir, cat_mod_ske_dir, item, modname))
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
                             transforms.Scale(size =  224, interpolation = 2),
                             transforms.ToTensor(),
                             # normalize,
                        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
                                         transforms.RandomCrop(127),
                                         transforms.RandomHorizontalFlip(),
                            ])
        self.validating = transforms.Compose([
                        transforms.CenterCrop(127),
                        ])

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()
        self.perCatValueMeter_metro = {}
        for item in self.cat:
            self.perCatValueMeter_metro[item] = AverageValueMeter()
        self.transformsb = transforms.Compose([
                             transforms.Scale(size =  224, interpolation = 2),
                        ])
        ###for get_allviews
        self.data_index = 0

    def __getitem__(self, index):
        cat_mod_img_dir, cat_mod_ske_dir, item, modname = self.datapath[index]
        if self.rotate:
            # load image
            para_path = os.path.join(cat_mod_img_dir, "rendering_metadata.txt")
            if self.train:
                N = np.random.randint(0, self.num_views)
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%N))
                im = self.dataAugmentation(im) #random crop
                params = open(para_path).readlines()[int(N)]
                azimuth, elevation, _, distance, _ = map(float, params.strip().split())
            else:
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%self.idx))
                im = self.validating(im)#center crop
                params = open(para_path).readlines()[int(self.idx)]
                azimuth, elevation, _, distance, _ = map(float, params.strip().split())
            R = camera_rotation(azimuth, elevation, distance)
            data = self.transforms(im)#scale
            data = data[:3, :, :]

            all_ske = h5py.File(os.path.join(cat_mod_ske_dir, SKE_FILENAME), 'r')
            point_set_line, point_set_square, point_set_skeleton = all_ske['line'][:], all_ske['square'][:], all_ske['skeleton'][:]
            point_set_line = random_sample_pointset(point_set_line, self.npoints_line)
            point_set_square = random_sample_pointset(point_set_square, self.npoints_skeleton)
            point_set_skeleton = random_sample_pointset(point_set_skeleton[:5000], self.npoints_skeleton)
            point_set_line = torch.from_numpy(point_set_line.dot(R.T)).type(torch.FloatTensor).contiguous()
            point_set_square = torch.from_numpy(point_set_square.dot(R.T)).type(torch.FloatTensor).contiguous()
            point_set_skeleton = torch.from_numpy(point_set_skeleton.dot(R.T)).type(torch.FloatTensor).contiguous()
            rotation = torch.from_numpy(R).type(torch.FloatTensor).contiguous()
            return data, rotation, point_set_skeleton, point_set_line, point_set_square, item, modname
        else:
            #load skeletal points
            all_ske = h5py.File(os.path.join(cat_mod_ske_dir, SKE_FILENAME), 'r')
            point_set_line, point_set_square, point_set_skeleton = all_ske['line'][:], all_ske['square'][:], all_ske['skeleton'][:]
            point_set_line = random_sample_pointset(point_set_line, self.npoints_line)
            point_set_square = random_sample_pointset(point_set_square, self.npoints_skeleton)
            point_set_skeleton = random_sample_pointset(point_set_skeleton[:5000], self.npoints_skeleton)
            point_set_line = torch.from_numpy(point_set_line).type(torch.FloatTensor).contiguous()
            point_set_square = torch.from_numpy(point_set_square).type(torch.FloatTensor).contiguous()
            point_set_skeleton = torch.from_numpy(point_set_skeleton).type(torch.FloatTensor).contiguous()
            # load image
            if self.train:
                N = np.random.randint(0, self.num_views)
                if self.white_bg:
                    img = cv2.imread(os.path.join(cat_mod_img_dir, "%02d.png"%N), cv2.IMREAD_UNCHANGED)
                    img[np.where(img[:,:,3]==0)] = 255
                    im = Image.fromarray(img)
                else:
                    im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%N))
                im = self.dataAugmentation(im) #random crop
            else:
                if self.white_bg:
                    img = cv2.imread(os.path.join(cat_mod_img_dir, "%02d.png"%self.idx), cv2.IMREAD_UNCHANGED)
                    img[np.where(img[:,:,3]==0)] = 255
                    im = Image.fromarray(img)
                else:
                    im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%self.idx))
                im = self.validating(im)#center crop
            data = self.transforms(im)#scale
            data = data[:3, :, :]

            return data, point_set_skeleton, point_set_line, point_set_square, item, modname

    def get_allviews(self):
        if self.data_index >= len(self.datapath):
            self.data_index = 0
        cat_mod_img_dir, cat_mod_ske_dir, item, modname = self.datapath[self.data_index]
        self.data_index += 1

        if self.rotate:
            #load skeletal points
            all_ske = h5py.File(os.path.join(cat_mod_ske_dir, SKE_FILENAME), 'r')
            point_set_line, point_set_square, point_set_skeleton = all_ske['line'][:], all_ske['square'][:], all_ske['skeleton'][:]
            point_set_line = random_sample_pointset(point_set_line, self.npoints_line)
            point_set_square = random_sample_pointset(point_set_square, self.npoints_skeleton)
            point_set_skeleton = random_sample_pointset(point_set_skeleton[:5000], self.npoints_skeleton)

            #load image
            data = []
            all_R = []
            all_point_set_skeleton = []
            all_point_set_line = []
            all_point_set_square = []
            para_path = os.path.join(cat_mod_img_dir, "rendering_metadata.txt")
            for idx in range(self.num_views):
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%idx))
                im = self.validating(im)
                data.append(self.transforms(im)[:3, :, :])
                params = open(para_path).readlines()[idx]
                azimuth, elevation, _, distance, _ = map(float, params.strip().split())
                R = camera_rotation(azimuth, elevation, distance)
                all_point_set_line.append(torch.from_numpy(point_set_line.dot(R.T)).type(torch.FloatTensor).contiguous())
                all_point_set_square.append(torch.from_numpy(point_set_square.dot(R.T)).type(torch.FloatTensor).contiguous())
                all_point_set_skeleton.append(torch.from_numpy(point_set_skeleton.dot(R.T)).type(torch.FloatTensor).contiguous())
                all_R.append(torch.from_numpy(R).type(torch.FloatTensor).contiguous())
            data = torch.stack(data, dim=0)
            point_set_skeleton = torch.stack(all_point_set_skeleton, dim=0).contiguous()
            point_set_line = torch.stack(all_point_set_line, dim=0).contiguous()
            point_set_square = torch.stack(all_point_set_square, dim=0).contiguous()
            rotation = torch.stack(all_R, dim=0).contiguous()
            return data, rotation, point_set_skeleton, point_set_line, point_set_square, item, modname
        else:
            #load skeletalall_ske = h5py.File(os.path.join(cat_mod_ske_dir, SKE_FILENAME), 'r')
            all_ske = h5py.File(os.path.join(cat_mod_ske_dir, SKE_FILENAME), 'r')
            point_set_line, point_set_square, point_set_skeleton = all_ske['line'][:], all_ske['square'][:], all_ske['skeleton'][:]
            point_set_line = random_sample_pointset(point_set_line, self.npoints_line)
            point_set_square = random_sample_pointset(point_set_square, self.npoints_skeleton)
            point_set_skeleton = random_sample_pointset(point_set_skeleton[:5000], self.npoints_skeleton)
            #point_set_skeleton = point_set_skeleton[:self.npoints_skeleton]
            point_set_skeleton = torch.from_numpy(point_set_skeleton).type(torch.FloatTensor).contiguous()
            point_set_line = torch.from_numpy(point_set_line).type(torch.FloatTensor).contiguous()
            point_set_square = torch.from_numpy(point_set_square).type(torch.FloatTensor).contiguous()

            #load image
            data = []
            for idx in range(self.num_views):
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%idx))
                im = self.validating(im)
                data.append(self.transforms(im)[:3, :, :])
            data = torch.stack(data, dim=0)
            point_set_skeleton = point_set_skeleton[None, :, :].repeat(self.num_views, 1, 1).contiguous()
            point_set_line = point_set_line[None, :, :].repeat(self.num_views, 1, 1).contiguous()
            point_set_square = point_set_square[None, :, :].repeat(self.num_views, 1, 1).contiguous()
            return data, point_set_skeleton, point_set_line, point_set_square, item, modname

    def __len__(self):
        return len(self.datapath)



if __name__  == '__main__':

    print('Testing Shapenet dataset')
    d  =  ShapeNet(class_choice =  "chair", balanced= False, train=True, npoints_line = 5000, SVR=True)
    a = len(d)
    d  =  ShapeNet(class_choice =  "chair", balanced= False, train=False, npoints_square = 5000, SVR=True)
    a = a + len(d)
    for i,data in enumerate(d,0):
        print(data[0],data[1],data[2],data[3],data[4])
    print(a)
