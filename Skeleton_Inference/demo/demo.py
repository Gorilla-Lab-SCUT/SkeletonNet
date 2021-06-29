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
parser.add_argument('--model_ske', type=str, default = '', help='CurSkeNet,SurSkeNet')
parser.add_argument('--model_vol', type=str, default = '', help='Volume Refinment Net')
parser.add_argument('--model', type=str, default = '', help='whole network')
parser.add_argument('--super_points', type=int, default = 5000,  help='number of skeleton points')
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
parser.add_argument('--category', type=str, default='all')
parser.add_argument('--load_lowres_only', action='store_true', default=False)
parser.add_argument('--load_highres_patch', action='store_true', default=False)
parser.add_argument('--guidance_only64', action='store_true', default=False)
parser.add_argument('--load_sperate', action='store_true', default=False)

parser.add_argument('--th', type=float, default=0.3, help='the thresold to compute IoU')
parser.add_argument('--save_vox_h5', action='store_true')
parser.add_argument('--save_mesh', action='store_true')
parser.add_argument('--save_ske', action='store_true')
parser.add_argument('--inpimg', type=str, default = './demo/images.png')
parser.add_argument('--outdir', type=str, default = './demo/results')
parser.add_argument('--woptfeat', action='store_true')
opt = parser.parse_args()
print (opt)

#define initail lines and squares
grid1, lines_array, lines_adjacent_tensor = define_lines(num_points=opt.num_points_line, nb_primitives=opt.nb_primitives_line)
grid2, faces_array, vertex_adj_matrix_tensor = define_squares(num_points=opt.num_points_square, nb_primitives=opt.nb_primitives_square)

if opt.load_sperate:
    network_ske = SVR_CurSur(pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
        nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square, bottleneck_size=opt.bottleneck_size)
    network_ske.apply(weights_init)
    network_ske.cuda()
    network_ske.load_state_dict(torch.load(opt.model_ske))
    print(" Previous skeleton weight loaded ")
    
    network_vol = Hierarchical_Refinement(num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, \
        global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
    network_vol.apply(weights_init)
    network_vol.cuda()
    network_vol.load_state_dict(torch.load(opt.model_vol))
    print(" Previous volume weight loaded ")

#Create network
network = ImgToVolume( pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
    bottleneck_size = opt.bottleneck_size, nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square,
    num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
network.grid1 = grid1
network.grid2 = grid2
network.lines_array = lines_array
network.faces_array = faces_array
network.apply(weights_init)
network.cuda()
if opt.load_sperate:
    network.img2skeleton = network_ske
    network_ske.cpu()
    network.skeleton2volume = network_vol
    network_vol.cpu()

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
network = network.cuda()

def fetch_data(inpimg):

    data_transforms = transforms.Compose([
                                 transforms.Scale(size =  224, interpolation = 2),
                                 transforms.ToTensor(),
                                 # normalize,
                                 ])
    data_validating = transforms.Compose([
                            transforms.CenterCrop(127),])

    im = Image.open(inpimg)
    im = data_validating(im)
    data = data_transforms(im)
    data = data[None, :3, :, :]
    point_set_skeleton = torch.zeros((1, opt.super_points, 3)).float()
    point_set_line = torch.zeros((1, opt.super_points, 3)).float()
    point_set_square = torch.zeros((1, opt.super_points, 3)).float()
    vol32 = torch.zeros((1, 32, 32, 32)).long()
    vol64 = torch.zeros((1, 64, 64, 64)).long()
    vol128 = torch.zeros((1, 128, 128, 128)).long()
    vol256 = torch.zeros((1, 256, 256, 256)).long()
    return data, point_set_skeleton, point_set_line, point_set_square, vol32, vol64, vol128, vol256

if opt.save_mesh or opt.save_ske or opt.save_vox_h5:
    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)

network.eval()
with torch.no_grad():
    t0=time.time()
    inpimg = opt.inpimg
    filename = inpimg.split('/')[-1]
    filename = filename[:-4] #remove .png
    cat, mod, idx = filename.split('_')
    img, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256 = fetch_data(inpimg)
    indices_array = []
    for bidx in range(img.size(0)):
        choices = np.arange(opt.patch_num)
        indices_array.append(choices)
    indices_array = np.stack(indices_array, axis=0)

    img = img.cuda()
    points_line = points_line.cuda()
    points_square = points_square.cuda()
    voxelGT32 = voxelGT32.cuda()
    voxelGT64 = voxelGT64.cuda()
    voxelGT128 = voxelGT128.cuda()
    voxelGT256 = voxelGT256.cuda()

    pointsReconstructed_cur, pointsReconstructed_sur, occupany32, occupany64, occupancy_patch36, occupancy_patch72 = network(img, indices_array)
    pointsReconstructed = torch.cat([pointsReconstructed_cur, pointsReconstructed_sur], 1)
    prediction128, batch_iou128 = eval_iou_res128(1, occupancy_patch36, voxelGT128, opt.th)
    prediction256, batch_iou256, batch_pre, batch_rec = eval_iou_pre_rec_res256(1, occupancy_patch72, voxelGT256, opt.th)

    if opt.save_vox_h5:
        catmod_out_dir_vox = os.path.join(outdir, 'SkeVolume', cat, mod)
        if not os.path.exists(catmod_out_dir_vox):
            os.makedirs(catmod_out_dir_vox)
        outfile64 = os.path.join(catmod_out_dir_vox, '64_max_fill.h5')
        outfile128 = os.path.join(catmod_out_dir_vox, '128_max_fill.h5')
        outfile256 = os.path.join(catmod_out_dir_vox, '256_max_fill.h5')
        prediction64 = F.softmax(occupany64, dim=1)
        prediction64 = torch.ge(prediction64[:, 1, :, :, :], opt.th).type(torch.cuda.FloatTensor)
        _, prediction64_numpy = holefill_cpu(prediction64[0].data.cpu().numpy())
        _, prediction128_numpy = holefill_cpu(prediction128[0].data.cpu().numpy())
        _, prediction256_numpy = holefill_cpu(prediction256[0].data.cpu().numpy())
        save_voxel_h5py(prediction64_numpy, outfile64)
        save_voxel_h5py(prediction128_numpy, outfile128)
        save_voxel_h5py(prediction256_numpy, outfile256)
    if opt.save_mesh:
        save_volume_obj(outdir, cat, mod+'_'+idx, prediction256[0], voxelGT256[0], holefill=True)
    if opt.save_ske:
        save_skeleton_ply(outdir, cat, mod+'_Cur', pointsReconstructed_cur[0])
        save_skeleton_ply(outdir, cat, mod+'_Sur',  pointsReconstructed_sur[0])
        save_skeleton_ply(outdir, cat, mod+'_All', pointsReconstructed[0])
    print(opt.inpimg, time.time()-t0)