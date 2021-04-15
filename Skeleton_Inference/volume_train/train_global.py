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
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()

sys.path.append('./auxiliary/')
from dataset import *
from model import *
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

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default= 50, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '', help='whole network')
parser.add_argument('--model_ske', type=str, default = '', help='CurSkeNet,SurSkeNet')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of skeleton points')
parser.add_argument('--bottleneck_size', type=int, default=512, help='bottleneck_size')
parser.add_argument('--num_points_line', type=int, default = 600,  help='number of curve points')
parser.add_argument('--nb_primitives_line', type=int, default = 20,  help='number of primitives')
parser.add_argument('--num_points_square', type=int, default = 2000,  help='number of sheet points')
parser.add_argument('--nb_primitives_square', type=int, default = 20,  help='number of primitives')
parser.add_argument('--samples_line', type=int, default = 4000,  help='number of sampled points in liness')
parser.add_argument('--samples_triangle', type=int, default = 24000,  help='number of sampled points in triangle')
parser.add_argument('--category', type=str, default='all/chair')
parser.add_argument('--env', type=str, default ="Im2Ske_Global"   ,  help='visdom env')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lrDecay', type=float, default=0.1)
parser.add_argument('--lrStep', type=float, default=100)
parser.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')
parser.add_argument('--load_lowres_only', action='store_true', default=True)
parser.add_argument('--guidance_only64', action='store_true')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--woptfeat', action='store_true')
opt = parser.parse_args()
print (opt)

#Launch visdom for visualization
vis = visdom.Visdom(port=8000 , env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat() + opt.env #now.isoformat()
if opt.rotate:
    dir_name = os.path.join('./checkpoints/%s_rotate/%s' % (opt.category, save_path))
else:
    dir_name = os.path.join('./checkpoints/%s/%s' % (opt.category, save_path))
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10
best_val_ce_loss = 10

# Create train/test dataloader on new views and test dataset on new models
if opt.category == "all":
    category_select = None
else:
    category_select = (opt.category).split('_')
dataset = ShapeNet(train=True, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points,\
        load_lowres_only=opt.load_lowres_only, rotate=opt.rotate)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, drop_last=True,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(train=False, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points,\
        load_lowres_only=opt.load_lowres_only, rotate=opt.rotate)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))

cudnn.benchmark = True
len_dataset = len(dataset)

#define initail lines and squares
grid1, lines_array, lines_adjacent_tensor = define_lines(num_points=opt.num_points_line, nb_primitives=opt.nb_primitives_line)
grid2, faces_array, vertex_adj_matrix_tensor = define_squares(num_points=opt.num_points_square, nb_primitives=opt.nb_primitives_square)

network_ske = SVR_CurSur(pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
    nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square, bottleneck_size=opt.bottleneck_size)
network_ske.apply(weights_init)
network_ske.cuda()
if opt.model_ske != '':
    network_ske.load_state_dict(torch.load(opt.model_ske))
    print(" Previous skeleton weight loaded ")
print(network_ske)

#Create network
network = Global_Refinement(num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous volume weight loaded ")
print(network)

#Create Loss Module
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
lrate = opt.lr
params_dict = dict(network.named_parameters())
params = []
optimizer = optim.Adam([{'params':network.parameters(), 'lr': opt.lr, 'weight_decay':opt.weight_decay}])

num_batch = len(dataset) / opt.batchSize
train_loss = AverageValueMeter()
loss1 = AverageValueMeter()
loss2 = AverageValueMeter()
loss3 = AverageValueMeter()
val_loss = AverageValueMeter()
val_loss1 = AverageValueMeter()
val_loss2 = AverageValueMeter()
val_loss3 = AverageValueMeter()
with open(logname, 'a') as f: #open and append
    f.write(str(network) + '\n')

train_curve = []
train_curve1 = []
train_curve2 = []
train_curve3 = []
val_curve = []
val_curve1 = []
val_curve2 = []
val_curve3 = []

trainloss_acc0 = 1e-9
trainloss_accs = 0
validloss_acc0 = 1e-9
validloss_accs = 0
for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    loss1.reset()
    loss2.reset()
    loss3.reset()
    network_ske.eval()
    network.train()
    if (epoch+1)%opt.lrStep == 0:
        optimizer = optim.Adam([{'params':network.parameters(), 'lr': lrate * opt.lrDecay, 'weight_decay':opt.weight_decay},])
        lrate = lrate * opt.lrDecay
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        if opt.rotate:
            img, rotation, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
            rotation = rotation.cuda()
        else:
            img, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
        img = img.cuda()
        points_line = points_line.cuda()
        points_square = points_square.cuda()
        voxelGT32 = voxelGT32.cuda()
        voxelGT64 = voxelGT64.cuda()

        pointsReconstructed_cur, pointsReconstructed_sur = network_ske.forward_inference(img, grid1, grid2)
        if opt.rotate:
            pointsReconstructed_cur = torch.bmm(pointsReconstructed_cur, rotation)
            pointsReconstructed_sur = torch.bmm(pointsReconstructed_sur, rotation)
        recon1, recon2, _ = network(img, pointsReconstructed_cur, lines_array, pointsReconstructed_sur, faces_array)
        if opt.guidance_only64:
            ce2 = criterion(recon2, voxelGT64)
            loss2.update(ce2.item())
            loss_net = ce2
            print('Ref64 loss : %f' %(ce2.item()))
        else:
            ce1 = criterion(recon1, voxelGT32)
            ce2 = criterion(recon2, voxelGT64)
            loss1.update(ce1.item())
            loss2.update(ce2.item())
            loss_net = ce1 + ce2
            print('Ref32 loss : %f, Ref64 loss : %f' %((ce1.item(), ce2.item())))
        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        optimizer.step()
        # VIZUALIZE
        print('[%d: %d/%d] train loss:  %f , %f ' %(epoch, i, len(dataset)/opt.batchSize, loss_net.item(), trainloss_accs/trainloss_acc0))

    # UPDATE CURVES
    train_curve.append(train_loss.avg)
    train_curve1.append(loss1.avg)
    train_curve2.append(loss2.avg)

    #VALIDATION
    val_loss.reset()
    val_loss1.reset()
    val_loss2.reset()
    val_loss3.reset()
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    network_ske.eval()
    network.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            if opt.rotate:
                img, rotation, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
                rotation = rotation.cuda()
            else:
                img, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
            img = img.cuda()
            points_line = points_line.cuda()
            points_square = points_square.cuda()
            voxelGT32 = voxelGT32.cuda()
            voxelGT64 = voxelGT64.cuda()

            pointsReconstructed_cur, pointsReconstructed_sur = network_ske.forward_inference(img, grid1, grid2)
            if opt.rotate:
                pointsReconstructed_cur = torch.bmm(pointsReconstructed_cur, rotation)
                pointsReconstructed_sur = torch.bmm(pointsReconstructed_sur, rotation)
            recon1, recon2, _ = network(img, pointsReconstructed_cur, lines_array, pointsReconstructed_sur, faces_array)
            if opt.guidance_only64:
                ce2 = criterion(recon2, voxelGT64)
                loss_net = ce2
                val_loss2.update(ce2.item())
                print('Ref64 loss : %f' %(ce2.item()))
            else:
                ce1 = criterion(recon1, voxelGT32)
                ce2 = criterion(recon2, voxelGT64)
                loss_net = ce1 + ce2
                val_loss1.update(ce1.item())
                val_loss2.update(ce2.item())
                print('Ref32 loss : %f, Ref64 loss : %f' %((ce1.item(), ce2.item())))
            validloss_accs = validloss_accs * 0.99 + loss_net.item()
            validloss_acc0 = validloss_acc0 * 0.99 + 1 
            val_loss.update(loss_net.item())
            dataset_test.perCatValueMeter[cat[0]].update(ce2.item())
            print('[%d: %d/%d] valid loss:  %f , %f ' %(epoch, i, len(dataset_test)/opt.batchSize, loss_net.item(), validloss_accs/validloss_acc0))
    #UPDATE CURVES
    val_curve.append(val_loss.avg)
    val_curve1.append(val_loss1.avg)
    val_curve2.append(val_loss2.avg)

    #UPDATE CURVES
    vis.line(
        X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
        Y=np.log(np.column_stack((np.array(train_curve), np.array(val_curve)))),
        win='All log',
        opts=dict(title="All loss", legend=["train_curve" + opt.env, "val_curve" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve1)), np.arange(len(val_curve1)))),
        Y=np.log(np.column_stack((np.array(train_curve1), np.array(val_curve1)))),
        win='Ref32 loss log',
        opts=dict(title="AE32 loss", legend=["train_curve1" + opt.env, "val_curve1" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve2)), np.arange(len(val_curve2)))),
        Y=np.log(np.column_stack((np.array(train_curve2), np.array(val_curve2)))),
        win='Ref64 loss log',
        opts=dict(title="Ref32 loss", legend=["train_curve2" + opt.env, "val_curve2" + opt.env],
                  markersize=2, ), )

    #save mini total loss
    if best_val_loss > val_loss.avg:
        best_val_loss = val_loss.avg
        print('New best loss : ', best_val_loss)
        print('saving net...')
        torch.save(network.state_dict(), '%s/network_epoch%d.pth' % (dir_name,epoch))
    #save mini bce loss
    if best_val_ce_loss > val_loss2.avg:
        best_val_ce_loss = val_loss2.avg
        print('New best bce loss : ', best_val_ce_loss)
        print('saving net...')
        torch.save(network.state_dict(), '%s/network_ce64_epoch%d.pth' % (dir_name,epoch))
    torch.save(network.state_dict(), '%s/network_last.pth' % dir_name)

    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "bestval" : best_val_loss,
      "bestval_ce" : best_val_ce_loss
    }
    print(log_table)

    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n