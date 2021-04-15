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
from PIL import Image
from config import ROOT_IMG, ROOT_PC, ROOT_VOL, ROOT_SPLIT, SYNSET_PATH, \
                    RENDER_DIRNAME, SKE_FILENAME,TRAIN_SPLIT_FILENAME, TEST_SPLIT_FILENAME
from utils import *

class ShapeNet(data.Dataset):
    def __init__(self, rootimg = ROOT_IMG, rootpc = ROOT_PC, rootvol = ROOT_VOL, class_choice = "chair",
                 npoints_skeleton=2500, npoints_line = 2500, npoints_square = 5000,
                 balanced = False, train = True, SVR=True, num_views=6, gen_view=False, idx=0,
                 load_lowres_only=False, load_highres_patch=False, holefill=True,
                 npatch=4, padding1=2, padding2=4, patch_res1=36, patch_res2=72, rotate=False):
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.rootvol = rootvol
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
        #add for skeletal volume load
        self.load_lowres_only = load_lowres_only
        self.load_highres_patch = load_highres_patch
        self.holefill = holefill
        self.npatch = npatch
        self.padding1 = padding1
        self.padding2 = padding2
        self.patch_res1 = patch_res1
        self.patch_res2 = patch_res2
        self.rotate = rotate
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
            dir_vol = os.path.join(self.rootvol, self.cat[item])
            #fns_ske = sorted(os.listdir(dir_ske))
            #fns = [val for val in fns if val + '.npz' in fns_ske and val in fns_img]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns)/float(len(fns_img)), "%")

            if len(fns) !=0:
                self.meta[item] = []
            for modname in fns:
                cat_mod_img_dir = os.path.join(dir_img, modname, RENDER_DIRNAME)
                cat_mod_ske_dir = os.path.join(dir_ske, modname)
                cat_mod_vol_dir = os.path.join(dir_vol, modname)
                self.meta[item].append((cat_mod_img_dir, cat_mod_ske_dir, cat_mod_vol_dir, item, modname))
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
        self.perCatValueMeter2 = {}
        self.perCatValueMeter_metro = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()
            self.perCatValueMeter2[item] = AverageValueMeter()
            self.perCatValueMeter_metro[item] = AverageValueMeter()
        self.transformsb = transforms.Compose([
                             transforms.Scale(size =  224, interpolation = 2),
                        ])
        ###for get_allviews
        self.data_index = 0

    def __getitem__(self, index):
        cat_mod_img_dir, cat_mod_ske_dir, cat_mod_vol_dir, item, modname = self.datapath[index]

        #load skeletal points
        ske_h5 = h5py.File(os.path.join(cat_mod_ske_dir, SKE_FILENAME), 'r')
        point_set_line, point_set_square, point_set_skeleton = ske_h5['line'][:], ske_h5['square'][:], ske_h5['skeleton'][:]
        point_set_line = random_sample_pointset(point_set_line, self.npoints_line)
        point_set_square = random_sample_pointset(point_set_square, self.npoints_skeleton)
        point_set_skeleton = random_sample_pointset(point_set_skeleton[:5000], self.npoints_skeleton)
        #point_set_skeleton = point_set_skeleton[:self.npoints_skeleton]

        #load skeletal volume
        vol32 = h5py.File(os.path.join(cat_mod_vol_dir, '32_fill.h5'), 'r')['occupancies'][:]
        vol64 = h5py.File(os.path.join(cat_mod_vol_dir, '64_max_fill.h5'), 'r')['occupancies'][:]
        if not self.load_lowres_only:
            vol128 = h5py.File(os.path.join(cat_mod_vol_dir, '128_max_fill.h5'), 'r')['occupancies'][:]
            vol256 = h5py.File(os.path.join(cat_mod_vol_dir, '256_max_fill.h5'), 'r')['occupancies'][:]

        vol32 = torch.from_numpy(vol32.astype('f4').reshape(32, 32, 32)).type(torch.LongTensor)
        vol64 = torch.from_numpy(vol64.astype('f4').reshape(64, 64, 64)).type(torch.LongTensor)
        if not self.load_lowres_only:
            vol128 = torch.from_numpy(vol128.astype('f4').reshape(128, 128, 128)).type(torch.LongTensor)
            vol256 = torch.from_numpy(vol256.astype('f4').reshape(256, 256, 256)).type(torch.LongTensor)
            if self.load_highres_patch:
                patches1 = []
                patches2 = []
                if self.padding1!=0:
                    vol128_padd = torch.zeros((128+2*self.padding1, 128+2*self.padding1, 128+2*self.padding1), dtype=vol128.dtype)
                    vol128_padd[self.padding1:-self.padding1, self.padding1:-self.padding1, self.padding1:-self.padding1] = vol128
                else:
                    vol128_padd = vol128
                if self.padding2!=0:
                    vol256_padd = torch.zeros((256+2*self.padding2, 256+2*self.padding2, 256+2*self.padding2), dtype=vol256.dtype)
                    vol256_padd[self.padding2:-self.padding2, self.padding2:-self.padding2, self.padding2:-self.padding2] = vol256
                else:
                    vol256_padd = vol256
                for i in range(self.npatch):
                    for j in range(self.npatch):
                        for k in range(self.npatch):
                            x1, x2 = i*32, i*32+self.patch_res1
                            y1, y2 = j*32, j*32+self.patch_res1
                            z1, z2 = k*32, k*32+self.patch_res1
                            patches1.append(vol128_padd[None, x1:x2, y1:y2, z1:z2])
                            x3, x4 = i*64, i*64+self.patch_res2
                            y3, y4 = j*64, j*64+self.patch_res2
                            z3, z4 = k*64, k*64+self.patch_res2
                            patches2.append(vol256_padd[None, x3:x4, y3:y4, z3:z4])
                vol128 = torch.cat(patches1, 0).contiguous()
                vol256 = torch.cat(patches2, 0).contiguous()
            else:
                pass

        if self.rotate:
            para_path = os.path.join(cat_mod_img_dir, "rendering_metadata.txt")
            # load image
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
            #rotate points
            point_set_line = torch.from_numpy(point_set_line.dot(R.T)).type(torch.FloatTensor).contiguous()
            point_set_square = torch.from_numpy(point_set_square.dot(R.T)).type(torch.FloatTensor).contiguous()
            point_set_skeleton = torch.from_numpy(point_set_skeleton.dot(R.T)).type(torch.FloatTensor).contiguous()
            rotation = torch.from_numpy(R).type(torch.FloatTensor).contiguous()
        else:
            # load image
            if self.train:
                N = np.random.randint(0, self.num_views)
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%N))
                im = self.dataAugmentation(im) #random crop
            else:
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%self.idx))
                im = self.validating(im)#center crop
            data = self.transforms(im)#scale
            data = data[:3, :, :]
            point_set_line = torch.from_numpy(point_set_line).type(torch.FloatTensor).contiguous()
            point_set_square = torch.from_numpy(point_set_square).type(torch.FloatTensor).contiguous()
            point_set_skeleton = torch.from_numpy(point_set_skeleton).type(torch.FloatTensor).contiguous()

        if not self.load_lowres_only:
            if self.rotate:
                return data, rotation, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, vol128, vol256, item, modname
            else:
                return data, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, vol128, vol256, item, modname
        else:
            if self.rotate:
                return data, rotation, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, 'vol128', 'vol256', item, modname
            else:
                return data, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, 'vol128', 'vol256', item, modname

    def get_allviews(self):
        if self.data_index >= len(self.datapath):
            self.data_index = 0
        cat_mod_img_dir, cat_mod_ske_dir, cat_mod_vol_dir, item, modname = self.datapath[self.data_index]
        self.data_index += 1

        ske_h5 = h5py.File(os.path.join(cat_mod_ske_dir, SKE_FILENAME), 'r')
        point_set_line, point_set_square, point_set_skeleton = ske_h5['line'][:], ske_h5['square'][:], ske_h5['skeleton'][:]
        point_set_line = random_sample_pointset(point_set_line, self.npoints_line)
        point_set_square = random_sample_pointset(point_set_square, self.npoints_skeleton)
        point_set_skeleton = random_sample_pointset(point_set_skeleton[:5000], self.npoints_skeleton)
        #point_set_skeleton = point_set_skeleton[:self.npoints_skeleton]

        #load skeletal volu
        vol32 = h5py.File(os.path.join(cat_mod_vol_dir, '32_fill.h5'), 'r')['occupancies'][:]
        vol64 = h5py.File(os.path.join(cat_mod_vol_dir, '64_max_fill.h5'), 'r')['occupancies'][:]
        if not self.load_lowres_only:
            vol128 = h5py.File(os.path.join(cat_mod_vol_dir, '128_max_fill.h5'), 'r')['occupancies'][:]
            vol256 = h5py.File(os.path.join(cat_mod_vol_dir, '256_max_fill.h5'), 'r')['occupancies'][:]

        vol32 = torch.from_numpy(vol32.astype('f4').reshape(32, 32, 32)).type(torch.LongTensor)
        vol64 = torch.from_numpy(vol64.astype('f4').reshape(64, 64, 64)).type(torch.LongTensor)
        if not self.load_lowres_only:
            vol128 = torch.from_numpy(vol128.astype('f4').reshape(128, 128, 128)).type(torch.LongTensor)
            vol256 = torch.from_numpy(vol256.astype('f4').reshape(256, 256, 256)).type(torch.LongTensor)
            if self.load_highres_patch:
                patches1 = []
                patches2 = []
                if self.padding1!=0:
                    vol128_padd = torch.zeros((128+2*self.padding1, 128+2*self.padding1, 128+2*self.padding1), dtype=vol128.dtype)
                    vol128_padd[self.padding1:-self.padding1, self.padding1:-self.padding1, self.padding1:-self.padding1] = vol128
                else:
                    vol128_padd = vol128
                if self.padding2!=0:
                    vol256_padd = torch.zeros((256+2*self.padding2, 256+2*self.padding2, 256+2*self.padding2), dtype=vol256.dtype)
                    vol256_padd[self.padding2:-self.padding2, self.padding2:-self.padding2, self.padding2:-self.padding2] = vol256
                else:
                    vol256_padd = vol256
                for i in range(self.npatch):
                    for j in range(self.npatch):
                        for k in range(self.npatch):
                            x1, x2 = i*32, i*32+self.patch_res1
                            y1, y2 = j*32, j*32+self.patch_res1
                            z1, z2 = k*32, k*32+self.patch_res1
                            patches1.append(vol128_padd[None, x1:x2, y1:y2, z1:z2])
                            x3, x4 = i*64, i*64+self.patch_res2
                            y3, y4 = j*64, j*64+self.patch_res2
                            z3, z4 = k*64, k*64+self.patch_res2
                            patches2.append(vol256_padd[None, x3:x4, y3:y4, z3:z4])
                vol128 = torch.cat(patches1, 0).contiguous()
                vol256 = torch.cat(patches2, 0).contiguous()
            else:
                pass

        if self.rotate:
            #load image
            data = []
            all_R = []
            all_point_set_skeleton = []
            all_point_set_line = []
            all_point_set_square = []
            para_path = os.path.join(cat_mod_img_dir, "rendering_metadata.txt")
            all_params = open(para_path).readlines()
            for idx in range(self.num_views):
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%idx))
                im = self.validating(im)
                data.append(self.transforms(im)[:3, :, :])
                params = all_params[idx]
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
        else:
            data = []
            for idx in range(self.num_views):
                im = Image.open(os.path.join(cat_mod_img_dir, "%02d.png"%idx))
                im = self.validating(im)
                data.append(self.transforms(im)[:3, :, :])
            data = torch.stack(data, dim=0)
            point_set_skeleton = torch.from_numpy(point_set_skeleton).type(torch.FloatTensor).contiguous()
            point_set_line = torch.from_numpy(point_set_line).type(torch.FloatTensor).contiguous()
            point_set_square = torch.from_numpy(point_set_square).type(torch.FloatTensor).contiguous()
            point_set_skeleton = point_set_skeleton[None, :, :].contiguous()
            point_set_line = point_set_line[None, :, :].contiguous()
            point_set_square = point_set_square[None, :, :].contiguous()
        vol32 = vol32[None, ...].contiguous()
        vol64 = vol64[None, ...].contiguous()
        vol128 = vol128[None, ...].contiguous()
        vol256 = vol256[None, ...].contiguous()
        if not self.load_lowres_only:
            if self.rotate:
                return data, rotation, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, vol128, vol256, item, modname
            else:
                return data, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, vol128, vol256, item, modname
        else:
            if self.rotate:
                return data, rotation, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, vol128, vol256, item, modname
            else:
                return data, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, 'vol128', 'vol256', item, modname

    def __len__(self):
        return len(self.datapath)

if __name__  == '__main__':

    print('Testing Shapenet dataset')
    d  =  ShapeNet(class_choice =  "chair", balanced= False, train=True, npoints_line = 5000, SVR=True)
    a = len(d)
    d  =  ShapeNet(class_choice =  "chair", balanced= False, train=False, npoints_square = 5000, SVR=True)
    a = a + len(d)
    for i,data in enumerate(d,0):
        print(data)
    print(a)