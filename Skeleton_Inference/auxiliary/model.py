from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time
#import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
import sys
from ske_utils import *
from model_pc import *
from model_voxel import *
from scipy import stats
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

####grid random selection
def shuffle_points_feat(pts, feat):
    batchsize = feat.size()[0]
    N = feat.size()[1]
    C = feat.size()[2]
    for k in range(batchsize):
        points_coord, points_feat = pts[k, :, :], feat[k, :, :]
        random_idx = torch.randperm(N)
        points_coord, points_feat = points_coord[random_idx, :], points_feat[random_idx, :]
        pts[k, :, :], feat[k, :, :] = points_coord, points_feat
    return pts.contiguous(), feat.contiguous()

def point_to_cell(pts, feat, W, H, D):
    pts = pts.type(torch.cuda.LongTensor)
    batchsize = feat.size()[0]
    N = feat.size()[1]
    C = feat.size()[2]

    feat_cell = torch.zeros((batchsize, W, H, D, C), dtype=feat.dtype, device=feat.device)
    for k in range(batchsize):
        points_coord, points_feat = pts[k, :, :], feat[k, :, :]
        x, y, z = points_coord[:, 0], points_coord[:, 1], points_coord[:,2]
        feat_cell[k, x, y, z, :] = points_feat
    feat_cell = feat_cell.reshape(batchsize, -1, C)
    feat_cell = torch.transpose(feat_cell, 1, 2).contiguous().view(-1, C, W, H, D)
    return feat_cell
####
####
def crop_global_local_patches(indices_array, occupany64_softmax_padding, feat_cell128_padding, glo_mul, glo_pch_res, loc_mul, loc_pch_res):
    global_patches, local_patches = [], []
    for b_idx in range(indices_array.shape[0]):
        for p_idx in range(indices_array.shape[1]):
            patch_idx = indices_array[b_idx, p_idx]
            i, j, k = int(patch_idx/16), int((patch_idx%16)/4), int(patch_idx%4)
            glo_x_start, glo_x_end = int(i*glo_mul), int(i*glo_mul+glo_pch_res)
            glo_y_start, glo_y_end = int(j*glo_mul), int(j*glo_mul+glo_pch_res)
            glo_z_start, glo_z_end = int(k*glo_mul), int(k*glo_mul+glo_pch_res)
            global_patches.append(occupany64_softmax_padding[b_idx:b_idx+1, 0:, glo_x_start:glo_x_end, glo_y_start:glo_y_end, glo_z_start: glo_z_end])
            loc_x_start, loc_x_end = int(i*loc_mul), int(i*loc_mul+loc_pch_res)
            loc_y_start, loc_y_end = int(j*loc_mul), int(j*loc_mul+loc_pch_res)
            loc_z_start, loc_z_end = int(k*loc_mul), int(k*loc_mul+loc_pch_res)
            local_patches.append(feat_cell128_padding[b_idx:b_idx+1, 0:, loc_x_start:loc_x_end, loc_y_start:loc_y_end, loc_z_start:loc_z_end])
    global_patches = torch.cat(global_patches, 0).contiguous()
    local_patches = torch.cat(local_patches, 0).contiguous()
    return global_patches, local_patches
####
#### Padding occupancy64_softmax
def padding_occupancy64_softmax(occupany64_softmax, padding16):
    occupany64_softmax_padding = torch.zeros((occupany64_softmax.size(0), occupany64_softmax.size(1), \
        64+2*padding16, 64+2*padding16, 64+2*padding16), dtype=occupany64_softmax.dtype, device=occupany64_softmax.device)
    occupany64_softmax_padding[0:, 0:, padding16:-padding16, padding16:-padding16, padding16:-padding16] = occupany64_softmax
    return occupany64_softmax_padding
####
#### Padding featcell128
def padding_featcell128(feat_cell128, padding32):
    feat_cell128_padding = torch.zeros((feat_cell128.size(0), feat_cell128.size(1), \
        128+2*padding32, 128+2*padding32, 128+2*padding32), dtype=feat_cell128.dtype, device=feat_cell128.device)
    feat_cell128_padding[0:, 0:, padding32:-padding32, padding32:-padding32, padding32:-padding32] = feat_cell128
    return feat_cell128_padding
####
####Grid Pooling 32
def batch_grid_pooling32(x, feat_pt):
    cube32 = x * 32 + 16.5
    cube32 = torch.clamp(cube32, 0, 31)
    feat_cell32 = point_to_cell(cube32, feat_pt, 32, 32, 32)
    return feat_cell32
####
####Grid Pooling 64
def batch_grid_pooling64(x, feat_pt):
    cube64 = x * 64 + 32.5
    cube64 = torch.clamp(cube64, 0, 63)
    feat_cell64 = point_to_cell(cube64, feat_pt, 64, 64, 64)
    return feat_cell64
####Grid Pooling 128 with padding:
def batch_grid_pooling128(x, feat_pt, padding32):
    cube128 = x * 128 + 64.5
    cube128 = torch.clamp(cube128, 0, 127)
    feat_cell128 = point_to_cell(cube128, feat_pt, 128, 128, 128)
    if padding32 !=0:
        feat_cell128_padding = padding_featcell128(feat_cell128, padding32)
        return feat_cell128_padding
    else:
        return feat_cell128

class Corrosion_Refinement(nn.Module):
    """docstring for Corrosion_Refinement"""
    def __init__(self,  num_samples_line=2000, num_samples_triangle=26000):
        super(Corrosion_Refinement, self).__init__()
        self.num_samples_line = num_samples_line
        self.num_samples_triangle = num_samples_triangle
        self.local_network = nn.MaxPool3d(kernel_size=3, stride=1, padding=2)

    def forward(self, imgs, curves, lines_array, surfaces, faces_array, indices_array):
        #uniformly sample the points/features on the lines/triangles
        curve_samples= batch_sample_lines(curves, lines_array, num=self.num_samples_line)
        surface_samples = batch_sample_triangles(surfaces, faces_array, num=self.num_samples_triangle)
        #combine samples with raw
        x = torch.cat([curves, curve_samples, surfaces, surface_samples], 1).contiguous()
        feat_pt = torch.ones((x.size(0), x.size(1), 1), dtype=torch.float32, device=x.device)

        #grid_pooling 256 then padding
        cube256 = x * 256 + 128.5
        cube256 = torch.clamp(cube256, 0, 255)
        feat_cell256 = point_to_cell(cube256, feat_pt, 256, 256, 256)
        occupancy256 = self.local_network(feat_cell256)
        return x, occupancy256

####
class PointNetfeatLocal(nn.Module):
    """Learn point-wise feature in the beginning of the network
    with fully connected layers the same as PointNet, the fully
    connected layers are implemented as 1d convolution so that
    it is independent to the number of points
    """ 
    def __init__(self):
        super(PointNetfeatLocal, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 16, 1)
    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        pointfeat = x.transpose(1, 2)
        return pointfeat

###Refine patch by patch without any global guidance
class Patch_Refinement(nn.Module):
    """docstring for Patch_Refinement"""
    def __init__(self,  num_samples_line=2000, num_samples_triangle=26000, npatch=4, padding32=2):
        super(Patch_Refinement, self).__init__()
        self.num_samples_line = num_samples_line
        self.num_samples_triangle = num_samples_triangle
        self.npatch = npatch
        self.padding32 = padding32
        self.loc_mul = 128/npatch
        self.loc_pch_res = 128/npatch + 2*padding32
        if padding32!=0:
            self.local_network = Patch36To72()
        else:
            self.local_network = Patch32To64()
    def forward(self, imgs, curves, lines_array, surfaces, faces_array, indices_array):
        #uniformly sample the points/features on the lines/triangles
        curve_samples= batch_sample_lines(curves, lines_array, num=self.num_samples_line)
        surface_samples = batch_sample_triangles(surfaces, faces_array, num=self.num_samples_triangle)
        #combine samples with raw
        x = torch.cat([curves, curve_samples, surfaces, surface_samples], 1).contiguous()
        feat_pt = torch.ones((x.size(0), x.size(1), 1), dtype=torch.float32, device=x.device)
        x, feat_pt = shuffle_points_feat(x, feat_pt)

        #grid_pooling 128 then padding
        feat_cell128_padding = batch_grid_pooling128(x, feat_pt, self.padding32)
        local_patches = []
        for b_idx in range(indices_array.shape[0]):
            for p_idx in range(indices_array.shape[1]):
                patch_idx = indices_array[b_idx, p_idx]
                i, j, k = patch_idx/16, (patch_idx%16)/4, patch_idx%4
                loc_x_start, loc_x_end = i*self.loc_mul, i*self.loc_mul+self.loc_pch_res
                loc_y_start, loc_y_end = j*self.loc_mul, j*self.loc_mul+self.loc_pch_res
                loc_z_start, loc_z_end = k*self.loc_mul, k*self.loc_mul+self.loc_pch_res
                local_patches.append(feat_cell128_padding[b_idx:b_idx+1, :, loc_x_start:loc_x_end, loc_y_start:loc_y_end, loc_z_start:loc_z_end])
        local_patches = torch.cat(local_patches, 0).contiguous()
        occupancy_patch36, occupany_patch72 = self.local_network(local_patches)
        return occupancy_patch36, occupany_patch72

####add global refinement 64 and 32
class Global_Refinement(nn.Module):
    def __init__(self, num_samples_line=2000, num_samples_triangle=26000, global_only64=False, woptfeat=False):
        super(Global_Refinement,self).__init__()
        self.feat = PointNetfeatLocal()
        self.global_only64 = global_only64
        if global_only64:
            self.global_network = Global_Guidance_Only64()
        else:
            self.global_network = Global_Guidance()
        self.num_samples_line = num_samples_line
        self.num_samples_triangle = num_samples_triangle
        self.woptfeat = woptfeat

    def forward(self, imgs, curves, lines_array, surfaces, faces_array):
        curves_surfaces = torch.cat([curves, surfaces], 1).contiguous()
        feat_skeleton = self.feat(curves_surfaces)
        curve_feat = feat_skeleton[:, 0:curves.size(1), :]
        surface_feat = feat_skeleton[:, curves.size(1):curves_surfaces.size(1), :]

        #uniformly sample the points/features on the lines/triangles
        curve_samples, curve_sample_feats = batch_sample_lines_feats(curves, lines_array, curve_feat, num=self.num_samples_line)
        surface_samples, surface_sample_feats = batch_sample_triangles_feats(surfaces, faces_array, surface_feat, num=self.num_samples_triangle)
        #combine samples with raw
        x = torch.cat([curves, curve_samples, surfaces, surface_samples], 1).contiguous()
        feat_pt = torch.cat([curve_feat, curve_sample_feats, surface_feat, surface_sample_feats], 1).contiguous()
        x, feat_pt = shuffle_points_feat(x, feat_pt)

        if not self.global_only64:
            feat_cell32 = batch_grid_pooling32(x, feat_pt)
        feat_cell64 = batch_grid_pooling64(x, feat_pt)
        if not self.global_only64:
            occupany32, occupany64, occupany64_softmax = self.global_network(feat_cell32, feat_cell64)
        else:
            occupany32 = None
            occupany64, occupany64_softmax = self.global_network(feat_cell64)
        return occupany32, occupany64, occupany64_softmax

class Hierarchical_Refinement(nn.Module):
    def __init__(self, num_samples_line=2000, num_samples_triangle=26000, global_only64 = False, woptfeat=False, dist2prob=False, npatch=4, padding16=1, padding32=2):
        super(Hierarchical_Refinement,self).__init__()
        self.feat = PointNetfeatLocal()
        if global_only64:
            self.global_network = Global_Guidance_Only64()
        else:
            self.global_network = Global_Guidance()
        self.local_network = Local_Synthesis(woptfeat=woptfeat)
        self.num_samples_line = num_samples_line
        self.num_samples_triangle = num_samples_triangle
        self.global_only64 = global_only64
        self.woptfeat = woptfeat
        self.dist2prob = dist2prob

        self.npatch = npatch
        self.padding16 = padding16
        self.padding32 = padding32
        self.glo_mul = 64 / npatch #16
        self.loc_mul = 128 / npatch #32
        self.glo_pch_res = 64 / npatch + 2 * padding16  #18 
        self.loc_pch_res = 128 / npatch + 2 * padding32 #36

    def forward(self, imgs, curves, lines_array, surfaces, faces_array, indices_array):
        curves_surfaces = torch.cat([curves, surfaces], 1).contiguous()
        feat_skeleton = self.feat(curves_surfaces)
        curve_feat = feat_skeleton[:, 0:curves.size(1), :]
        surface_feat = feat_skeleton[:, curves.size(1):curves_surfaces.size(1), :]

        #uniformly sample the points/features on the lines/triangles
        curve_samples, curve_sample_feats = batch_sample_lines_feats(curves, lines_array, curve_feat, num=self.num_samples_line)
        surface_samples, surface_sample_feats = batch_sample_triangles_feats(surfaces, faces_array, surface_feat, num=self.num_samples_triangle)

        #combine samples with raw
        x = torch.cat([curves, curve_samples, surfaces, surface_samples], 1).contiguous()
        feat_pt = torch.cat([curve_feat, curve_sample_feats, surface_feat, surface_sample_feats], 1).contiguous()
        x, feat_pt = shuffle_points_feat(x, feat_pt)

        if not self.global_only64:
            feat_cell32 = batch_grid_pooling32(x, feat_pt)
        feat_cell64 = batch_grid_pooling64(x, feat_pt)
        if not self.global_only64:
            occupany32, occupany64, occupany64_softmax = self.global_network(feat_cell32, feat_cell64)
        else:
            occupany32 = None
            occupany64, occupany64_softmax = self.global_network(feat_cell64)
        ###
        if self.woptfeat:
            #### don't use pointfeat for high resolution refinement
            occupany64_softmax_padding = padding_occupancy64_softmax(occupany64_softmax[:, 1:2, ...], self.padding16)
            feat_pt = torch.ones((x.size(0), x.size(1), 1), dtype=torch.float32, device=x.device)
        else:
            occupany64_softmax_padding = padding_occupancy64_softmax(occupany64_softmax[:, 0:2, ...], self.padding16)
        ####
        if self.dist2prob:
            print('Have not implemented now!')
        else:
            feat_cell128_padding = batch_grid_pooling128(x, feat_pt, self.padding32)
        global_patches, local_patches = crop_global_local_patches(indices_array, occupany64_softmax_padding, feat_cell128_padding, \
                self.glo_mul, self.glo_pch_res, self.loc_mul, self.loc_pch_res)
        occupancy_patch36, occupany_patch72 = self.local_network(global_patches, local_patches)
        #return cur_up, sur_up, occupany32, occupany64, occupancy_patch36, occupany_patch72
        return occupany32, occupany64, occupancy_patch36, occupany_patch72

    def forward_globalGT_pretrain(self, imgs, curves, lines_array, surfaces, faces_array, indices_array, occupany64_gt):
        curves_surfaces = torch.cat([curves, surfaces], 1).contiguous()
        feat_skeleton = self.feat(curves_surfaces)
        curve_feat = feat_skeleton[:, 0:curves.size(1), :]
        surface_feat = feat_skeleton[:, curves.size(1):curves_surfaces.size(1), :]

        #uniformly sample the points/features on the lines/triangles
        curve_samples, curve_sample_feats = batch_sample_lines_feats(curves, lines_array, curve_feat, num=self.num_samples_line)
        surface_samples, surface_sample_feats = batch_sample_triangles_feats(surfaces, faces_array, surface_feat, num=self.num_samples_triangle)
        #combine samples with raw
        x = torch.cat([curves, curve_samples, surfaces, surface_samples], 1).contiguous()
        feat_pt = torch.cat([curve_feat, curve_sample_feats, surface_feat, surface_sample_feats], 1).contiguous()
        x, feat_pt = shuffle_points_feat(x, feat_pt)
        ####
        occupany64_gt_padding = padding_occupancy64_softmax(occupany64_gt, self.padding16)
        feat_pt = torch.ones((x.size(0), x.size(1), 1), dtype=torch.float32, device=x.device)
        ####
        feat_cell128_padding = batch_grid_pooling128(x, feat_pt, self.padding32)
        global_patches, local_patches = crop_global_local_patches(indices_array, occupany64_gt_padding, feat_cell128_padding, \
                self.glo_mul, self.glo_pch_res, self.loc_mul, self.loc_pch_res)
        occupancy_patch36, occupany_patch72 = self.local_network(global_patches, local_patches)
        return None, None, occupancy_patch36, occupany_patch72

class ImgToVolume(nn.Module):
    def __init__(self, pretrained_encoder=False, num_points_line = 2048, num_points_square = 2048, bottleneck_size = 512, nb_primitives_line = 1, nb_primitives_square = 1,\
            num_samples_line=2000, num_samples_triangle=26000, global_only64 = False, woptfeat=False, dist2prob=False):
        super(ImgToVolume, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.num_points_line = num_points_line
        self.num_points_square = num_points_square
        self.bottleneck_size = bottleneck_size
        self.nb_primitives_line = nb_primitives_line
        self.nb_primitives_square = nb_primitives_square
        self.num_samples_line = num_samples_line
        self.num_samples_triangle = num_samples_triangle
        self.global_only64 = global_only64
        self.woptfeat = woptfeat
        self.dist2prob = dist2prob
        self.grid1 = None
        self.grid2 = None
        self.lines_array = None
        self.faces_array = None
        self.img2skeleton = SVR_CurSur(pretrained_encoder = self.pretrained_encoder, num_points_line = self.num_points_line, num_points_square = self.num_points_square, 
            bottleneck_size = self.bottleneck_size, nb_primitives_line = self.nb_primitives_line, nb_primitives_square = self.nb_primitives_square)
        self.skeleton2volume = Hierarchical_Refinement(num_samples_line=num_samples_line, num_samples_triangle=self.num_samples_triangle, global_only64 = self.global_only64, woptfeat=self.woptfeat, dist2prob=self.dist2prob)

    def forward(self, x, indices_array):
        cur_ske, sur_ske = self.img2skeleton.forward_inference(x, self.grid1, self.grid2)
        occupany32, occupany64, occupancy_patch36, occupany_patch72 = self.skeleton2volume(x, cur_ske, self.lines_array, sur_ske, self.faces_array, indices_array)
        return cur_ske, sur_ske, occupany32, occupany64, occupancy_patch36, occupany_patch72

class ImgToVolume_Rotate(nn.Module):
    def __init__(self, pretrained_encoder=False, num_points_line = 2048, num_points_square = 2048, bottleneck_size = 512, nb_primitives_line = 1, nb_primitives_square = 1,\
            num_samples_line=2000, num_samples_triangle=26000, global_only64 = False, woptfeat=False, dist2prob=False):
        super(ImgToVolume_Rotate, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.num_points_line = num_points_line
        self.num_points_square = num_points_square
        self.bottleneck_size = bottleneck_size
        self.nb_primitives_line = nb_primitives_line
        self.nb_primitives_square = nb_primitives_square
        self.num_samples_line = num_samples_line
        self.num_samples_triangle = num_samples_triangle
        self.global_only64 = global_only64
        self.woptfeat = woptfeat
        self.dist2prob = dist2prob
        self.grid1 = None
        self.grid2 = None
        self.lines_array = None
        self.faces_array = None
        self.img2skeleton = SVR_CurSur(pretrained_encoder = self.pretrained_encoder, num_points_line = self.num_points_line, num_points_square = self.num_points_square, 
            bottleneck_size = self.bottleneck_size, nb_primitives_line = self.nb_primitives_line, nb_primitives_square = self.nb_primitives_square)
        self.skeleton2volume = Hierarchical_Refinement(num_samples_line=num_samples_line, num_samples_triangle=self.num_samples_triangle, global_only64 = self.global_only64, dist2prob=self.dist2prob)

    def forward(self, x, rotation, indices_array):
        cur_ske, sur_ske = self.img2skeleton.forward_inference(x, self.grid1, self.grid2)
        cur_ske_align = torch.bmm(cur_ske, rotation)
        sur_ske_align = torch.bmm(sur_ske, rotation)
        occupany32, occupany64, occupancy_patch36, occupany_patch72 = self.skeleton2volume(x, cur_ske_align, self.lines_array, sur_ske_align, self.faces_array, indices_array)
        return cur_ske, sur_ske, occupany32, occupany64, occupancy_patch36, occupany_patch72

if __name__ == '__main__':
    # print('testing GridPooling...')
    # points = Variable(torch.rand(1, 30000, 3).type(dtype), requires_grad=False) * 100.0
    # feat_pt = Variable(torch.rand(1, 30000, 8).type(dtype), requires_grad=True)
    # feat_cell = point_to_cell(points, feat_pt, 128, 128, 128)

    print('testing SkeletonToVolume network...')
    sim_data1 = Variable(torch.rand(1, 3, 2500))
    sim_data1 = sim_data1.cuda()
    sim_data2 = Variable(torch.rand(1, 3, 2500))
    sim_data2 = sim_data2.cuda()
    model = SkeletonToVolume()
    model.eval()
    model =  model.cuda()
    out1, out2, out3, out4 = model(sim_data1, sim_data2)
    print(out1.size(), out2.size(), out3.size(), out4.size())

    print('testing SurfaceToVolume network...')
    sim_data = Variable(torch.rand(1, 3, 2500))
    sim_data = sim_data.cuda()
    model = SurfaceToVolume()
    model.eval()
    model =  model.cuda()
    out1, out2, out3, out4 = model(sim_data)
    print(out1.size(), out2.size(), out3.size(), out4.size())

    print('testing ImgToVolume network...')
    sim_data = Variable(torch.rand(1, 3, 2500))
    sim_data = sim_data.cuda()
    model = ImgToVolume()
    model.eval()
    model =  model.cuda()
    out1, out2, out3, out4 = model(sim_data)
    print(out1.size(), out2.size(), out3.size(), out4.size())
