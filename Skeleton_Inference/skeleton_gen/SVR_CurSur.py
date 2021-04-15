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
sys.path.append('./auxiliary/')
from dataset_pc import *
from model_pc import *
from utils import *
from ske_utils import *
from plyio import *
import torch.nn.functional as F
import sys
from tqdm import tqdm
import os
import json
import time, datetime
import visdom

sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()
best_val_loss = 10

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--model_ae', type=str, default = '',  help='model path')
parser.add_argument('--num_points_line', type=int, default = 600,  help='number of curve points')
parser.add_argument('--nb_primitives_line', type=int, default = 20,  help='number of primitives')
parser.add_argument('--num_points_square', type=int, default = 2000,  help='number of sheet points')
parser.add_argument('--nb_primitives_square', type=int, default = 20,  help='number of primitives')
parser.add_argument('--bottleneck_size', type=int, default=512, help='bottleneck_size')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of input points to pointNet, not used by default')
parser.add_argument('--fix_decoder',  default = 'False'   ,  help='if set to True, on the pointNet encoder is trained')
parser.add_argument('--k1',type=float,default=0)
parser.add_argument('--category', type=str, default='all/chair')
parser.add_argument('--save_dir', type=str, default='./skeleton_gen/SVR_CurSur')
parser.add_argument('--rotate', action='store_true')
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
dataset = ShapeNet(train=True, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points, rotate=opt.rotate)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, drop_last=True,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(train=False, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points, rotate=opt.rotate)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
cudnn.benchmark = True
len_dataset = len(dataset)

#define initail lines and squares
grid1, lines_array, lines_adjacent_tensor = define_lines(num_points=opt.num_points_line, nb_primitives=opt.nb_primitives_line)
grid2, faces_array, vertex_adj_matrix_tensor = define_squares(num_points=opt.num_points_square, nb_primitives=opt.nb_primitives_square)

#Load Pretrained skeleton autoencoder
if opt.fix_decoder == 'True':
    network_ae = AE_CurSur(num_points =opt.super_points, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
        nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square, bottleneck_size=opt.bottleneck_size)
    network_ae.cuda()
    network_ae.load_state_dict(torch.load(opt.model_ae))
    network_ae.eval()

#Create network
network = SVR_CurSur(pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
    nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square, bottleneck_size=opt.bottleneck_size)
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

if opt.fix_decoder == 'True':
    network.decoder_line = network_ae.decoder_line
    network.decoder_square = network_ae.decoder_square
    network_ae.cpu()
network = network.cuda()
print(network)

outroot = opt.save_dir
if not os.path.exists(outroot):
    os.mkdir(outroot)
    print('creat dir',outroot)

val_loss = AverageValueMeter()
validloss_acc0 = 1e-9
validloss_accs = 0
val_loss.reset()
for item in dataset_test.cat:
    dataset_test.perCatValueMeter[item].reset()

network.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        if opt.rotate:
            img, rotation, points_skeleton, points_line, points_square, cat, mod = data
            rotation = rotation.cuda()
        else:
            img, points_skeleton, points_line, points_square, cat, mod = data
        img = img.cuda()
        points_skeleton = points_skeleton.cuda()
        points_line = points_line.cuda()
        points_square = points_square.cuda()
        GT = torch.cat([points_line, points_square], 1).contiguous()

        pointsReconstructed_cur, pointsReconstructed_sur = network.forward_inference(img, grid1, grid2)
        pointsReconstructed = torch.cat([pointsReconstructed_cur, pointsReconstructed_sur], 1).contiguous()
        cur_dist1, cur_dist2 = distChamfer(points_line, pointsReconstructed_cur)
        cur_laplacian_smooth = curve_laplacian(pointsReconstructed_cur, opt.nb_primitives_line, lines_adjacent_tensor)
        sur_dist1, sur_dist2 = distChamfer(points_square, pointsReconstructed_sur)
        sur_laplacian_smooth = surface_laplacian(pointsReconstructed_sur, opt.nb_primitives_square, vertex_adj_matrix_tensor)
        cd_total = torch.mean(cur_dist1) + torch.mean(cur_dist2) + torch.mean(sur_dist1) + torch.mean(sur_dist2)
        laplacian_total = torch.mean(cur_laplacian_smooth) + torch.mean(sur_laplacian_smooth)
        loss_net = cd_total + opt.k1* laplacian_total
        validloss_accs = validloss_accs * 0.99 + loss_net.item()
        validloss_acc0 = validloss_acc0 * 0.99 + 1 
        val_loss.update(loss_net.item())
        dataset_test.perCatValueMeter[cat[0]].update(cd_total.item())


        catname = cat[0]
        modname = mod[0]
        out_cat_dir = os.path.join(outroot, cats_dict[catname])
        if not os.path.exists(out_cat_dir):
            os.mkdir(out_cat_dir)
            os.mkdir(out_cat_dir+'/img')
        if opt.rotate:
            pointsReconstructed_cur = torch.bmm(pointsReconstructed_cur, rotation)
            pointsReconstructed_sur = torch.bmm(pointsReconstructed_sur, rotation)
            pointsReconstructed = torch.bmm(pointsReconstructed, rotation)
            points_line = torch.bmm(points_line, rotation)
            points_square = torch.bmm(points_square, rotation)
            GT = torch.bmm(GT, rotation)
        line_pred = pointsReconstructed_cur.cpu().data.squeeze().numpy()
        square_pred = pointsReconstructed_sur.cpu().data.squeeze().numpy()
        pred = pointsReconstructed.cpu().data.squeeze().numpy()
        line_gt = points_line.cpu().data.squeeze().numpy()
        square_gt = points_square.cpu().data.squeeze().numpy()
        gt = GT.cpu().data.squeeze().numpy()

        img_file = os.path.join(out_cat_dir, 'img/'+modname+'.png')
        line_pred_file = os.path.join(out_cat_dir, modname+'_line_pred.ply')
        square_pred_file = os.path.join(out_cat_dir, modname+'_square_pred.ply')
        pred_file = os.path.join(out_cat_dir, modname+'_pred.ply')
        line_gt_file = os.path.join(out_cat_dir, modname+'_line_gt.ply')
        square_gt_file = os.path.join(out_cat_dir, modname+'_square_gt.ply')
        gt_file = os.path.join(out_cat_dir, modname+'_gt.ply')

        img_inp = img.cpu().data.squeeze().numpy().transpose(1,2,0)
        cv2.imwrite(img_file, img_inp*255)
        write_ply(filename = line_pred_file, points=pd.DataFrame(line_pred), as_text=True)
        write_ply(filename = square_pred_file, points=pd.DataFrame(square_pred), as_text=True)
        write_ply(filename = pred_file, points=pd.DataFrame(pred), as_text=True)
        write_ply(filename = line_gt_file, points=pd.DataFrame(line_gt), as_text=True)
        write_ply(filename = square_gt_file, points=pd.DataFrame(square_gt), as_text=True)
        write_ply(filename = gt_file, points=pd.DataFrame(gt), as_text=True)
        print('[%d: %d/%d] valid loss:  %f , %f ' %(0, i, len(dataset_test)/opt.batchSize, loss_net.item(), validloss_accs/validloss_acc0))
