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
from plyio import *
import torch.nn.functional as F
import sys
from tqdm import tqdm
import os
import json
import time, datetime
import visdom

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default= 20, help='number of epochs to train for')
parser.add_argument('--start_val_epoch', type=int, default=5)
parser.add_argument('--model', type=str, default = '', help='whole network')
parser.add_argument('--model_glo', type=str, default = '', help='gloabl network')
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
parser.add_argument('--patch_num', type=int, default = 32, help='the number of patch for training')
parser.add_argument('--category', type=str, default='all/chair')
parser.add_argument('--env', type=str, default ="Im2Ske_Local"   ,  help='visdom env')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lrDecay', type=float, default=0.1)
parser.add_argument('--lrStep', type=float, default=100)
parser.add_argument('--weight_ce',type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')
parser.add_argument('--load_lowres_only', action='store_true', default=False)
parser.add_argument('--load_highres_patch', action='store_true', default=True)
parser.add_argument('--guidance_only64', action='store_true', default=False)
parser.add_argument('--globalGT_pretrain', action='store_true', default=False)
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--woptfeat', action='store_true')
opt = parser.parse_args()
print (opt)

#Launch visdom for visualization
vis = visdom.Visdom(port=8000 , env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat() + opt.env #now.isoformat() + 
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
        load_lowres_only=opt.load_lowres_only, load_highres_patch=opt.load_highres_patch, rotate=opt.rotate)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, drop_last=True,
                                         shuffle=True, num_workers=int(opt.workers))
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

network_ske = SVR_CurSur(pretrained_encoder=False, num_points_line = opt.num_points_line, num_points_square = opt.num_points_square,
    nb_primitives_line = opt.nb_primitives_line, nb_primitives_square = opt.nb_primitives_square, bottleneck_size=opt.bottleneck_size)
network_ske.apply(weights_init)
network_ske.cuda()
if opt.model_ske != '':
    network_ske.load_state_dict(torch.load(opt.model_ske))
    print(" Previous skeleton weight loaded ")
print(network_ske)

#Create network
network_global = Global_Refinement(num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
network_global.apply(weights_init)
network_global.cuda()
if opt.model_glo != '':
    network_global.load_state_dict(torch.load(opt.model_glo))
    print(" Previous volume weight loaded ")
print(network_global)

network = Hierarchical_Refinement(num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
network.apply(weights_init)
network.cuda()
if opt.model_glo !='':
    network.feat = network_global.feat
    network.global_network = network_global.global_network
    network_global.cpu()

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
network = network.cuda()
print(network)

#Create Loss Module
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_weight = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, opt.weight_ce]))
criterion_weight = criterion_weight.cuda()

lrate = opt.lr
params_dict = dict(network.named_parameters())
params = []
optimizer = optim.Adam([
    {'params':network.local_network.parameters(), 'lr':opt.lr, 'weight_decay':opt.weight_decay}])

num_batch = len(dataset) / opt.batchSize
train_loss = AverageValueMeter()
loss1 = AverageValueMeter()
loss2 = AverageValueMeter()
loss3 = AverageValueMeter()
loss4 = AverageValueMeter()
val_loss = AverageValueMeter()
val_loss1 = AverageValueMeter()
val_loss2 = AverageValueMeter()
val_loss3 = AverageValueMeter()
val_loss4 = AverageValueMeter()
with open(logname, 'a') as f: #open and append
    f.write(str(network) + '\n')

train_curve = []
train_curve1 = []
train_curve2 = []
train_curve3 = []
train_curve4 = []
val_curve = []
val_curve1 = []
val_curve2 = []
val_curve3 = []
val_curve4 = []

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
    loss4.reset()
    network_ske.eval()
    network.feat.eval()
    network.global_network.eval()
    network.local_network.train()
    if (epoch+1)%opt.lrStep == 0:
        optimizer = optim.Adam([
            {'params':network.local_network.parameters(), 'lr':lrate * opt.lrDecay, 'weight_decay':opt.weight_decay}])
        lrate = lrate * opt.lrDecay
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        t0 = time.time()
        if opt.rotate:
            img, rotation, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
            rotation = rotation.cuda()
        else:
            img, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
        fetch_time = time.time() - t0
        indices_array = []
        for bidx in range(img.size(0)):
            choices = np.arange(64)
            dist = stats.rv_discrete(name='custm', values=(choices, np.ones(64)/64.0))
            choices = dist.rvs(size=opt.patch_num)
            indices_array.append(choices)
        indices_array = np.stack(indices_array, axis=0) #[B, patch_num]
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

        pointsReconstructed_cur, pointsReconstructed_sur = network_ske.forward_inference(img, grid1, grid2)
        if opt.rotate:
            pointsReconstructed_cur = torch.bmm(pointsReconstructed_cur, rotation)
            pointsReconstructed_sur = torch.bmm(pointsReconstructed_sur, rotation)
        if opt.globalGT_pretrain:
            occupany32, occupany64, occupancy_patch36, occupany_patch72 = network.forward_globalGT_pretrain(img, pointsReconstructed_cur, lines_array, \
                pointsReconstructed_sur, faces_array, indices_array, voxelGT64.unsqueeze(1).type(torch.cuda.FloatTensor))
        else:
            occupany32, occupany64, occupancy_patch36, occupany_patch72 = network(img, pointsReconstructed_cur, lines_array, \
                pointsReconstructed_sur, faces_array, indices_array)
        ce3 = criterion_weight(occupancy_patch36, voxelGT128)
        ce4 = criterion_weight(occupany_patch72, voxelGT256)
        loss3.update(ce3.item())
        loss4.update(ce4.item())
        loss_net = ce3 + ce4
        print('Patch36 loss : %f, Patch72: %f' %((ce3.item(), ce4.item())))

        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        optimizer.step()
        # VIZUALIZE
        total_time = time.time() - t0
        print('[%d: %d/%d] train loss:  %f , %f , fetch: %f, total: %f' %(epoch, i, len(dataset)/opt.batchSize, loss_net.item(), trainloss_accs/trainloss_acc0, fetch_time, total_time))

    # UPDATE CURVES
    train_curve.append(train_loss.avg)
    train_curve1.append(loss1.avg)
    train_curve2.append(loss2.avg)
    train_curve3.append(loss3.avg)
    train_curve4.append(loss4.avg)

    #VALIDATION
    val_loss.reset()
    val_loss1.reset()
    val_loss2.reset()
    val_loss3.reset()
    val_loss4.reset()
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    if epoch>=opt.start_val_epoch:
        network_ske.eval()
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                t0 = time.time()
                if opt.rotate:
                    img, rotation, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
                    rotation = rotation.cuda()
                else:
                    img, points_skeleton, points_line, points_square, voxelGT32, voxelGT64, voxelGT128, voxelGT256, cat, mod = data
                fetch_time = time.time()-t0
                indices_array = []
                for bidx in range(img.size(0)):
                    choices = np.arange(64)
                    indices_array.append(choices)
                indices_array = np.stack(indices_array, axis=0)
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

                pointsReconstructed_cur, pointsReconstructed_sur = network_ske.forward_inference(img, grid1, grid2)
                if opt.rotate:
                    pointsReconstructed_cur = torch.bmm(pointsReconstructed_cur, rotation)
                    pointsReconstructed_sur = torch.bmm(pointsReconstructed_sur, rotation)
                if opt.globalGT_pretrain:
                    occupany32, occupany64, occupancy_patch36, occupany_patch72 = network.forward_globalGT_pretrain(img, pointsReconstructed_cur, lines_array, \
                        pointsReconstructed_sur, faces_array, indices_array, voxelGT64.unsqueeze(1).type(torch.cuda.FloatTensor))
                else:
                    occupany32, occupany64, occupancy_patch36, occupany_patch72 = network(img, pointsReconstructed_cur, lines_array, \
                        pointsReconstructed_sur, faces_array, indices_array)
                ce3 = criterion_weight(occupancy_patch36, voxelGT128)
                ce4 = criterion_weight(occupany_patch72, voxelGT256)
                val_loss3.update(ce3.item())
                val_loss4.update(ce4.item())
                loss_net = ce3 + ce4
                print('Patch36 loss : %f, Patch72: %f' %((ce3.item(), ce4.item())))
                
                validloss_accs = validloss_accs * 0.99 + loss_net.item()
                validloss_acc0 = validloss_acc0 * 0.99 + 1 
                val_loss.update(loss_net.item())
                dataset_test.perCatValueMeter[cat[0]].update(ce4.item())
                total_time = time.time()-t0
                print('[%d: %d/%d] valid loss:  %f , %f , fetch: %f, total: %f'%(epoch, i, len(dataset_test)/opt.batchSize, loss_net.item(), validloss_accs/validloss_acc0, fetch_time, total_time))

    #UPDATE CURVES
    val_curve.append(val_loss.avg)
    val_curve1.append(val_loss1.avg)
    val_curve2.append(val_loss2.avg)
    val_curve3.append(val_loss3.avg)
    val_curve4.append(val_loss4.avg)

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
        opts=dict(title="Ref32 loss", legend=["train_curve1" + opt.env, "val_curve1" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve2)), np.arange(len(val_curve2)))),
        Y=np.log(np.column_stack((np.array(train_curve2), np.array(val_curve2)))),
        win='Ref64 loss log',
        opts=dict(title="Ref64 loss", legend=["train_curve2" + opt.env, "val_curve2" + opt.env],
                  markersize=2, ), )
    vis.line(X=np.column_stack((np.arange(len(train_curve3)), np.arange(len(val_curve3)))),
                 Y=np.log(np.column_stack((np.array(train_curve3),np.array(val_curve3)))),
                 win='Pah36 loss log',
                 opts=dict(title="Pah36 loss", legend=["train_curve3" + opt.env, "val_curve3" + opt.env], 
                    markersize=2, ), )
    vis.line(X=np.column_stack((np.arange(len(train_curve4)), np.arange(len(val_curve4)))),
                 Y=np.log(np.column_stack((np.array(train_curve4),np.array(val_curve4)))),
                 win='Pah72 loss log',
                 opts=dict(title="Pah72 loss", legend=["train_curve4" + opt.env, "val_curve4" + opt.env], 
                    markersize=2, ), )

    if epoch>=opt.start_val_epoch:
        #save mini total loss
        if best_val_loss > val_loss.avg:
            best_val_loss = val_loss.avg
            print('New best loss : ', best_val_loss)
            print('saving net...')
            torch.save(network.state_dict(), '%s/network_bestall_epoch%d.pth' % (dir_name,epoch))
        #save mini bce loss
        if best_val_ce_loss > val_loss4.avg:
            best_val_ce_loss = val_loss4.avg
            print('New best bce loss : ', best_val_ce_loss)
            print('saving net...')
            torch.save(network.state_dict(), '%s/network_best256_epoch%d.pth' % (dir_name,epoch))
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
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

