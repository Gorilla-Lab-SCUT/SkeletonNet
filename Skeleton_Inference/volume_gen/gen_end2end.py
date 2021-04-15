from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
from scipy import stats
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()

sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ske_utils import *
from vox_utils import *
from plyio import *
import torch.nn.functional as F
import sys
from tqdm import tqdm
import os
import json
import time, datetime
import visdom

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--nepoch', type=int, default= 1, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '', help='whole network')
parser.add_argument('--model_vol', type=str, default = '', help='gloabl network')
parser.add_argument('--model_ske', type=str, default = '', help='CurSkeNet,SurSkeNet')

parser.add_argument('--super_points', type=int, default = 2500,  help='number of skeleton points')
parser.add_argument('--bottleneck_size', type=int, default=512, help='bottleneck_size')
parser.add_argument('--num_points_line', type=int, default = 600,  help='number of curve points')
parser.add_argument('--nb_primitives_line', type=int, default = 20,  help='number of primitives')
parser.add_argument('--num_points_square', type=int, default = 2000,  help='number of sheet points')
parser.add_argument('--nb_primitives_square', type=int, default = 20,  help='number of primitives')
parser.add_argument('--samples_line', type=int, default = 4000,  help='number of sampled points in liness')
parser.add_argument('--samples_triangle', type=int, default = 24000,  help='number of sampled points in triangle')

parser.add_argument('--patch_res1', type=int, default = 36, help='the resolution of patch')
parser.add_argument('--patch_res2', type=int, default = 72, help='the resolution of patch')
parser.add_argument('--patch_num', type=int, default = 64, help='the number of patch for training')
parser.add_argument('--category', type=str, default='all/chair')
parser.add_argument('--load_lowres_only', action='store_true', default=False)
parser.add_argument('--load_highres_patch', action='store_true', default=False)
parser.add_argument('--guidance_only64', action='store_true', default=False)

parser.add_argument('--th', type=float, default=0.3, help='the thresold to compute IoU')
parser.add_argument('--save_vox_h5', action='store_true')
parser.add_argument('--outdir', type=str, default = './volume_gen/local')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--woptfeat', action='store_true')
opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Create train/test dataloader on new views and test dataset on new models
if opt.category == "all":
    category_select = None
else:
    category_select = (opt.category).split('_')
dataset = ShapeNet(train=True, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points,\
        load_lowres_only=opt.load_lowres_only, load_highres_patch=opt.load_highres_patch, rotate=opt.rotate)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, 
                                         shuffle=False, num_workers=int(opt.workers))
dataset_test = ShapeNet(train=False, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points,\
        load_lowres_only=opt.load_lowres_only, load_highres_patch=opt.load_highres_patch, rotate=opt.rotate)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
cudnn.benchmark = True
len_dataset = len(dataset)

#define initail lines and squares
grid1, lines_array, lines_adjacent_tensor = define_lines(num_points=opt.num_points_line, nb_primitives=opt.nb_primitives_line)
grid2, faces_array, vertex_adj_matrix_tensor = define_squares(num_points=opt.num_points_square, nb_primitives=opt.nb_primitives_square)

if opt.load_sperate == "True":
    network_ske = SVR_CurSur(pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
        nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square, bottleneck_size=opt.bottleneck_size)
    network_ske.apply(weights_init)
    network_ske.cuda()
    if opt.model_ske != '':
        network_ske.load_state_dict(torch.load(opt.model_ske))
        print(" Previous skeleton weight loaded ")
    #print(network_ske)

    #Create network
    network_vol = Hierarchical_Refinement(num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, \
        global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
    network_vol.apply(weights_init)
    network_vol.cuda()
    if opt.model_vol != '':
        network_vol.load_state_dict(torch.load(opt.model_vol))
        print(" Previous volume weight loaded ")
    #print(network_vol)

#Create network
if opt.rotate:
    network = ImgToVolume_Rotate( pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
        bottleneck_size = opt.bottleneck_size, nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square,
        num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
else:
    network = ImgToVolume( pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
        bottleneck_size = opt.bottleneck_size, nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square,
        num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
network.grid1 = grid1
network.grid2 = grid2
network.lines_array = lines_array
network.faces_array = faces_array
network.apply(weights_init)
network.cuda()
if opt.load_sperate == "True":
    network.img2skeleton = network_ske
    network_ske.cpu()
    network.skeleton2volume = network_vol
    network_vol.cpu()

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
#Parallel 
#network = nn.DataParallel(network)
network = network.cuda()
print(network)
if opt.save_mesh or opt.save_ske or opt.save_vox_h5:
    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)

iou32 = dict()
iou64 = dict()
iou128 = dict()
iou256 = dict()
precision = dict()
recall = dict()
for item in dataset.cat:
    iou32[item] = AverageValueMeter()
    iou64[item] = AverageValueMeter()
    iou128[item] = AverageValueMeter()
    iou256[item] = AverageValueMeter()
    precision[item] = AverageValueMeter()
    recall[item] = AverageValueMeter()

iou_table32 = {item: iou32[item].avg}
iou_table64 = {item: iou64[item].avg}
iou_table128 = {item: iou128[item].avg}
iou_table256 = {item: iou256[item].avg}
pre_table = {item: precision[item].avg}
rec_table = {item: recall[item].avg}

network.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        t0 = time.time()
        if opt.rotate:
            img, rotation, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
            rotation = rotation.cuda()
        else:
            img, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
        ###
        indices_array = []
        for bidx in range(img.size(0)):
            choices = np.arange(opt.patch_num)
            indices_array.append(choices)
        indices_array = np.stack(indices_array, axis=0) #[B, 64]
        ####
        img = img.cuda()
        points_line = points_line.cuda()
        points_square = points_square.cuda()
        voxelGT32 = voxelGT32.cuda()
        voxelGT64 = voxelGT64.cuda()
        voxelGT128 = voxelGT128.cuda()
        voxelGT256 = voxelGT256.cuda()

        if opt.rotate:
            pointsReconstructed_cur, pointsReconstructed_sur, occupany32, occupany64, occupancy_patch36, occupancy_patch72 = network(img, rotation, indices_array)
            pointsReconstructed = torch.cat([pointsReconstructed_cur, pointsReconstructed_sur], 1)
            pointsReconstructed_cur = torch.bmm(pointsReconstructed_cur, rotation)
            pointsReconstructed_sur = torch.bmm(pointsReconstructed_sur, rotation)
            pointsReconstructed = torch.bmm(pointsReconstructed, rotation)
        else:
            pointsReconstructed_cur, pointsReconstructed_sur, occupany32, occupany64, occupancy_patch36, occupancy_patch72 = network(img, indices_array)
            pointsReconstructed = torch.cat([pointsReconstructed_cur, pointsReconstructed_sur], 1)
        pred32, batch_iou32 = eval_iou(occupany32, voxelGT32, opt.th)
        pred64, batch_iou64 = eval_iou(occupany64, voxelGT64, opt.th)
        prediction128, batch_iou128 = eval_iou_res128(len(cat), occupancy_patch36, voxelGT128, opt.th)
        prediction256, batch_iou256, batch_pre, batch_rec = eval_iou_pre_rec_res256(len(cat), occupancy_patch72, voxelGT256, opt.th)
        for j in range(len(cat)):
            iou32[cat[j]].update((batch_iou32[j]).item())
            iou64[cat[j]].update((batch_iou64[j]).item())
            iou128[cat[j]].update((batch_iou128[j]).item())
            iou256[cat[j]].update((batch_iou256[j]).item())
            precision[cat[j]].update((batch_pre[j]).item())
            recall[cat[j]].update((batch_rec[j].item()))
            catname = cat[j]
            modname = mod[j]
            if opt.save_vox_h5:
                catmod_out_dir = os.path.join(outdir, cat[j], mod[j])
                if not os.path.exists(catmod_out_dir):
                    os.makedirs(catmod_out_dir)
                outfile64 = os.path.join(catmod_out_dir, '64_max_fill.h5')
                outfile128 = os.path.join(catmod_out_dir, '128_max_fill.h5')
                outfile256 = os.path.join(catmod_out_dir, '256_max_fill.h5')
                _, prediction64_numpy = holefill_cpu(pred64[j].data.cpu().numpy())
                _, prediction128_numpy = holefill_cpu(prediction128[j].data.cpu().numpy())
                _, prediction256_numpy = holefill_cpu(prediction256[j].data.cpu().numpy())
                save_voxel_h5py(prediction64_numpy, outfile64)
                save_voxel_h5py(prediction128_numpy, outfile128)
                save_voxel_h5py(prediction256_numpy, outfile256)
            if opt.save_mesh:
                #save_mc_simplify_obj(outdir, catname, modname, prediction128[j], voxelGT128[j], holefill=True)
                save_mc_simplify_obj(outdir, catname, modname, prediction256[j], voxelGT256[j], holefill=True)

        for item in dataset_test.cat:
            iou_table32.update({item: iou32[item].avg})
            iou_table64.update({item: iou64[item].avg})
            iou_table128.update({item: iou128[item].avg})
            iou_table256.update({item: iou256[item].avg})
            pre_table.update({item: precision[item].avg})
            rec_table.update({item: recall[item].avg})
        print('TrainSet [%d/%d], time: %f' %(i, len(dataset)/opt.batchSize, time.time()-t0))
        print('Refine 32 iou', iou_table32)
        print('Refine 64 iou', iou_table64)
        print('Refine 128 iou', iou_table128)
        print('Refine 256 iou', iou_table256)
        print('Refine 256 pre', pre_table)
        print('Refine 256 rec', rec_table)



iou32 = dict()
iou64 = dict()
iou128 = dict()
iou256 = dict()
precision = dict()
recall = dict()
for item in dataset.cat:
    iou32[item] = AverageValueMeter()
    iou64[item] = AverageValueMeter()
    iou128[item] = AverageValueMeter()
    iou256[item] = AverageValueMeter()
    precision[item] = AverageValueMeter()
    recall[item] = AverageValueMeter()

iou_table32 = {item: iou32[item].avg}
iou_table64 = {item: iou64[item].avg}
iou_table128 = {item: iou128[item].avg}
iou_table256 = {item: iou256[item].avg}
pre_table = {item: precision[item].avg}
rec_table = {item: recall[item].avg}

network.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        t0 = time.time()
        if opt.rotate:
            img, rotation, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
            rotation = rotation.cuda()
        else:
            img, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
        ###
        indices_array = []
        for bidx in range(img.size(0)):
            choices = np.arange(opt.patch_num)
            indices_array.append(choices)
        indices_array = np.stack(indices_array, axis=0) #[B, 64]
        ####
        img = img.cuda()
        points_line = points_line.cuda()
        points_square = points_square.cuda()
        voxelGT32 = voxelGT32.cuda()
        voxelGT64 = voxelGT64.cuda()
        sample_GT128, sample_GT256 = [], []
        for bidx in range(img.size(0)):
            sample_GT128.append(voxelGT128[bidx, indices_array[bidx]])
            sample_GT256.append(voxelGT256[bidx, indices_array[bidx]])
        voxelGT128 = torch.cat(sample_GT128, dim=0).contiguous()
        voxelGT128 = voxelGT128.cuda()
        voxelGT256 = torch.cat(sample_GT256, dim=0).contiguous()
        voxelGT256 = voxelGT256.cuda()

        if opt.rotate:
            pointsReconstructed_cur, pointsReconstructed_sur, occupany32, occupany64, occupancy_patch36, occupancy_patch72 = network(img, rotation, indices_array)
            pointsReconstructed = torch.cat([pointsReconstructed_cur, pointsReconstructed_sur], 1)
            pointsReconstructed_cur = torch.bmm(pointsReconstructed_cur, rotation)
            pointsReconstructed_sur = torch.bmm(pointsReconstructed_sur, rotation)
            pointsReconstructed = torch.bmm(pointsReconstructed, rotation)
        else:
            pointsReconstructed_cur, pointsReconstructed_sur, occupany32, occupany64, occupancy_patch36, occupancy_patch72 = network(img, indices_array)
            pointsReconstructed = torch.cat([pointsReconstructed_cur, pointsReconstructed_sur], 1)
        pred32, batch_iou32 = eval_iou(occupany32, voxelGT32, opt.th)
        pred64, batch_iou64 = eval_iou(occupany64, voxelGT64, opt.th)
        prediction128, batch_iou128 = eval_iou_res128(len(cat), occupancy_patch36, voxelGT128, opt.th)
        prediction256, batch_iou256, batch_pre, batch_rec = eval_iou_pre_rec_res256(len(cat), occupancy_patch72, voxelGT256, opt.th)
        for j in range(len(cat)):
            iou32[cat[j]].update((batch_iou32[j]).item())
            iou64[cat[j]].update((batch_iou64[j]).item())
            iou128[cat[j]].update((batch_iou128[j]).item())
            iou256[cat[j]].update((batch_iou256[j]).item())
            precision[cat[j]].update((batch_pre[j]).item())
            recall[cat[j]].update((batch_rec[j].item()))
            catname = cat[j]
            modname = mod[j]
            if opt.save_vox_h5:
                catmod_out_dir = os.path.join(outdir, cat[j], mod[j])
                if not os.path.exists(catmod_out_dir):
                    os.makedirs(catmod_out_dir)
                outfile64 = os.path.join(catmod_out_dir, '64_max_fill.h5')
                outfile128 = os.path.join(catmod_out_dir, '128_max_fill.h5')
                outfile256 = os.path.join(catmod_out_dir, '256_max_fill.h5')
                _, prediction64_numpy = holefill_cpu(pred64[j].data.cpu().numpy())
                _, prediction128_numpy = holefill_cpu(prediction128[j].data.cpu().numpy())
                _, prediction256_numpy = holefill_cpu(prediction256[j].data.cpu().numpy())
                save_voxel_h5py(prediction64_numpy, outfile64)
                save_voxel_h5py(prediction128_numpy, outfile128)
                save_voxel_h5py(prediction256_numpy, outfile256)
            if opt.save_mesh:
                #save_mc_simplify_obj(outdir, catname, modname, prediction128[j], voxelGT128[j], holefill=True)
                save_mc_simplify_obj(outdir, catname, modname, prediction256[j], voxelGT256[j], holefill=True)

        for item in dataset_test.cat:
            iou_table32.update({item: iou32[item].avg})
            iou_table64.update({item: iou64[item].avg})
            iou_table128.update({item: iou128[item].avg})
            iou_table256.update({item: iou256[item].avg})
            pre_table.update({item: precision[item].avg})
            rec_table.update({item: recall[item].avg})
        print('TestSet[%d/%d], time: %f' %(i, len(dataset_test)/opt.batchSize, time.time()-t0))
        print('Refine 32 iou', iou_table32)
        print('Refine 64 iou', iou_table64)
        print('Refine 128 iou', iou_table128)
        print('Refine 256 iou', iou_table256)
        print('Refine 256 pre', pre_table)
        print('Refine 256 rec', rec_table)