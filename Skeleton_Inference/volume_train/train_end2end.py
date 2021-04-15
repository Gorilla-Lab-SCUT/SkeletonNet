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
parser.add_argument('--nepoch', type=int, default= 10, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '', help='whole network')
parser.add_argument('--model_ske', type=str, default = '', help='CurSkeNet,SurSkeNet')
parser.add_argument('--model_vol', type=str, default = '', help='whole network')
parser.add_argument('--model_glo', type=str, default = '', help='global network')
parser.add_argument('--load_sperate', action='store_true', default=False)
parser.add_argument('--set_bn_eval', action='store_true', default=False)

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
parser.add_argument('--env', type=str, default ="Im2Ske_end2end"   ,  help='visdom env')
parser.add_argument('--lr1', type=float, default=1e-4)
parser.add_argument('--lr2', type=float, default=1e-4)
parser.add_argument('--lrDecay', type=float, default=0.1)
parser.add_argument('--lrStep', type=float, default=100)
parser.add_argument('--start_eval_epoch', type=float, default=20)
parser.add_argument('--k1', type=float, default=0.2)
parser.add_argument('--weight_pts',type=float, default=1.0)
parser.add_argument('--weight_vox', type=float, default=1.0)
parser.add_argument('--weight_ce',type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')
parser.add_argument('--load_lowres_only', action='store_true', default=False)
parser.add_argument('--load_highres_patch', action='store_true', default=True)
parser.add_argument('--guidance_only64', action='store_true', default=False)
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
best_val_cd_loss = 10
best_val_bce_all_loss = 10
best_val_bce_final_loss = 10

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
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                          shuffle=False, num_workers=1)
print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
cudnn.benchmark = True
len_dataset = len(dataset)

#define initail lines and squares
grid1, lines_array, lines_adjacent_tensor = define_lines(num_points=opt.num_points_line, nb_primitives=opt.nb_primitives_line)
grid2, faces_array, vertex_adj_matrix_tensor = define_squares(num_points=opt.num_points_square, nb_primitives=opt.nb_primitives_square)

if opt.load_sperate:
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

    if opt.model_glo!= '':
        network_global = Global_Refinement(num_samples_line=opt.samples_line, num_samples_triangle=opt.samples_triangle, \
            global_only64=opt.guidance_only64, woptfeat=opt.woptfeat)
        network_global.apply(weights_init)
        network_global.cuda()
        network_global.load_state_dict(torch.load(opt.model_glo))
        print(" Previous global volume weight loaded ")
        #print(network_global)

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
if opt.load_sperate:
    network.img2skeleton = network_ske
    network_ske.cpu()
    network.skeleton2volume = network_vol
    network_vol.cpu()
    if opt.model_glo!='':
        network.skeleton2volume.feat = network_global.feat
        network.skeleton2volume.global_network = network_global.global_network

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
#Parallel 
#network = nn.DataParallel(network)
network = network.cuda()
print(network)

#Create Loss Module
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_weight = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, opt.weight_ce]))
criterion_weight = criterion_weight.cuda()

lrate1 = opt.lr1
lrate2 = opt.lr2
params_dict = dict(network.named_parameters())
params = []
optimizer = optim.Adam([
    {'params':network.img2skeleton.encoder.parameters(), 'lr': lrate1},
    {'params':network.skeleton2volume.parameters(), 'lr': lrate2, 'weight_decay':opt.weight_decay},
    ])

num_batch = len(dataset) / opt.batchSize
train_loss = AverageValueMeter()
train_loss1 = AverageValueMeter()
train_loss2 = AverageValueMeter()
train_loss3 = AverageValueMeter()
train_loss4 = AverageValueMeter()
train_loss5 = AverageValueMeter()
train_loss6 = AverageValueMeter()
train_loss7 = AverageValueMeter()
val_loss = AverageValueMeter()
val_loss1 = AverageValueMeter()
val_loss2 = AverageValueMeter()
val_loss3 = AverageValueMeter()
val_loss4 = AverageValueMeter()
val_loss5 = AverageValueMeter()
val_loss6 = AverageValueMeter()
val_loss7 = AverageValueMeter()
val_loss8 = AverageValueMeter()
with open(logname, 'a') as f: #open and append
    f.write(str(network) + '\n')

train_curve = []
train_curve1 =[]
train_curve2 = []
train_curve3 = []
train_curve4 = []
train_curve5 = []
train_curve6 = []
train_curve7 = []
val_curve = []
val_curve1 = []
val_curve2 = []
val_curve3 = []
val_curve4 = []
val_curve5 = []
val_curve6 = []
val_curve7 = []
val_curve8 = []
trainloss_acc0 = 1e-9
trainloss_accs = 0
validloss_acc0 = 1e-9
validloss_accs = 0

for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    train_loss1.reset()
    train_loss2.reset()
    train_loss3.reset()
    train_loss4.reset()
    train_loss5.reset()
    train_loss6.reset()
    train_loss7.reset()
    network.train()
    if opt.set_bn_eval:
        network.img2skeleton.apply(set_bn_eval)
    if (epoch+1)%opt.lrStep == 0:
        lrate1 = lrate1 * opt.lrDecay
        lrate2 = lrate2 * opt.lrDecay
        optimizer = optim.Adam([
                        {'params':network.img2skeleton.encoder.parameters(), 'lr': lrate1},
                        {'params':network.skeleton2volume.parameters(), 'lr': lrate2, 'weight_decay':opt.weight_decay},
                    ])
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
            distribution = np.ones(64)/64.0
            dist = stats.rv_discrete(name='custm', values=(choices, distribution))
            choices = dist.rvs(size=opt.patch_num)
            indices_array.append(choices)
        indices_array = np.stack(indices_array, axis=0) #[B, patch_num]
        img = img.cuda()
        points_skeleton = points_skeleton.cuda()
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
            pointsReconstructed_cur, pointsReconstructed_sur, recon1, recon2, recon3, recon4 = network(img, rotation, indices_array)
        else:
            pointsReconstructed_cur, pointsReconstructed_sur, recon1, recon2, recon3, recon4 = network(img, indices_array)
        cur_dist1, cur_dist2 = distChamfer(points_line, pointsReconstructed_cur)
        cur_laplacian_smooth = curve_laplacian(pointsReconstructed_cur, opt.nb_primitives_line, lines_adjacent_tensor)
        sur_dist1, sur_dist2 = distChamfer(points_square, pointsReconstructed_sur)
        sur_laplacian_smooth = surface_laplacian(pointsReconstructed_sur, opt.nb_primitives_square, vertex_adj_matrix_tensor)
        cd_total = torch.mean(cur_dist1) + torch.mean(cur_dist2) + torch.mean(sur_dist1) + torch.mean(sur_dist2)
        laplacian_total = torch.mean(cur_laplacian_smooth) + torch.mean(sur_laplacian_smooth)

        ce1 = criterion(recon1, voxelGT32)
        ce2 = criterion(recon2, voxelGT64)
        ce3 = criterion_weight(recon3, voxelGT128)
        ce4 = criterion_weight(recon4, voxelGT256)
        ce = ce1 + ce2 + ce3 + ce4

        loss_net = (cd_total + laplacian_total* opt.k1) * opt.weight_pts + ce * opt.weight_vox
        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        train_loss1.update(ce1.item())
        train_loss2.update(ce2.item())
        train_loss3.update(ce3.item())
        train_loss4.update(ce4.item())
        train_loss5.update(cd_total.item())
        train_loss6.update(laplacian_total.item())
        train_loss7.update(ce.item())
        optimizer.step()
        # VIZUALIZE
        if i%10 <= 0:
            vis.scatter(X = points_line[0].data.cpu(),
                win = 'REAL_TRAIN_CURVE',
                opts = dict(
                    title = "REAL_TRAIN_CURVE",
                    markersize = 2,
                    ),
                )
            vis.scatter(X = pointsReconstructed_cur[0].data.cpu(),
                win = 'FAKE_TRAIN_CURVE',
                opts = dict(
                    title="FAKE_TRAIN_CURVE",
                    markersize=2,
                    ),
                )
            vis.scatter(X = points_square[0].data.cpu(),
                win = 'REAL_TRAIN_SHEET',
                opts = dict(
                    title = "REAL_TRAIN_SHEET",
                    markersize = 2,
                    ),
                )
            vis.scatter(X = pointsReconstructed_sur[0].data.cpu(),
                win = 'FAKE_TRAIN_SHEET',
                opts = dict(
                    title="FAKE_TRAIN_SHEET",
                    markersize=2,
                    ),
                )
        total_time = time.time()-t0
        print('[%d: %d/%d] train loss:  %f , %f , fetch: %f, total: %f' %(epoch, i, len(dataset)/opt.batchSize, loss_net.item(), trainloss_accs/trainloss_acc0, fetch_time, total_time))
        print('CD loss : %f ; laplacian loss : %f, CrossEntropy : %f' %((cd_total.item(), laplacian_total.item(), ce.item())))
        print('CE1 : %f, CE2 : %f, CE3 : %f, CE4 : %f' %(ce1.item(), ce2.item(), ce3.item(), ce4.item()))

    #UPDATE CURVES
    train_curve.append(train_loss.avg)
    train_curve1.append(train_loss1.avg)
    train_curve2.append(train_loss2.avg)
    train_curve3.append(train_loss3.avg)
    train_curve4.append(train_loss4.avg)
    train_curve5.append(train_loss5.avg)
    train_curve6.append(train_loss6.avg)
    train_curve7.append(train_loss7.avg)

    if epoch>=opt.start_eval_epoch:
        #VALIDATION
        val_loss.reset()
        val_loss1.reset()
        val_loss2.reset()
        val_loss3.reset()
        val_loss4.reset()
        val_loss5.reset()
        val_loss6.reset()
        val_loss7.reset()
        val_loss8.reset()
        for item in dataset_test.cat:
            dataset_test.perCatValueMeter[item].reset()
            dataset_test.perCatValueMeter2[item].reset()

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
                ###
                indices_array = []
                for bidx in range(img.size(0)):
                    choices = np.arange(64)
                    indices_array.append(choices)
                indices_array = np.stack(indices_array, axis=0) #[B, patch_num]
                ###
                img = img.cuda()
                points_skeleton = points_skeleton.cuda()
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
                    pointsReconstructed_cur, pointsReconstructed_sur, recon1, recon2, recon3, recon4 = network(img, rotation, indices_array)
                else:
                    pointsReconstructed_cur, pointsReconstructed_sur, recon1, recon2, recon3, recon4 = network(img, indices_array)
                cur_dist1, cur_dist2 = distChamfer(points_line, pointsReconstructed_cur)
                cur_laplacian_smooth = curve_laplacian(pointsReconstructed_cur, opt.nb_primitives_line, lines_adjacent_tensor)
                sur_dist1, sur_dist2 = distChamfer(points_square, pointsReconstructed_sur)
                sur_laplacian_smooth = surface_laplacian(pointsReconstructed_sur, opt.nb_primitives_square, vertex_adj_matrix_tensor)
                cd_total = torch.mean(cur_dist1) + torch.mean(cur_dist2) + torch.mean(sur_dist1) + torch.mean(sur_dist2)
                laplacian_total = torch.mean(cur_laplacian_smooth) + torch.mean(sur_laplacian_smooth)
                ####
                pointsReconstructed = torch.cat((pointsReconstructed_cur, pointsReconstructed_sur), dim=1).contiguous()
                dist1, dist2 = distChamfer(points_skeleton, pointsReconstructed)
                real_cd = torch.mean(dist1) + torch.mean(dist2)
                ####
                ce1 = criterion(recon1, voxelGT32)
                ce2 = criterion(recon2, voxelGT64)
                ce3 = criterion_weight(recon3, voxelGT128)
                ce4 = criterion_weight(recon4, voxelGT256)
                ce = ce1 + ce2 + ce3 + ce4

                loss_net = (cd_total + laplacian_total* opt.k1)*opt.weight_pts + ce * opt.weight_vox
                validloss_accs = validloss_accs * 0.99 + loss_net.item()
                validloss_acc0 = validloss_acc0 * 0.99 + 1 
                val_loss.update(loss_net.item())
                val_loss1.update(ce1.item())
                val_loss2.update(ce2.item())
                val_loss3.update(ce3.item())
                val_loss4.update(ce4.item())
                val_loss5.update(cd_total.item())
                val_loss6.update(laplacian_total.item())
                val_loss7.update(ce.item())
                val_loss8.update(real_cd.item())
                dataset_test.perCatValueMeter[cat[0]].update(real_cd.item())
                dataset_test.perCatValueMeter2[cat[0]].update(ce4.item())

                if i%5 ==0 :
                    vis.scatter(X = points_line[0].data.cpu(),
                        win = 'REAL_TRAIN_CURVE',
                            opts = dict(
                            title = "REAL_TRAIN_CURVE",
                            markersize = 2,
                            ),
                        )
                    vis.scatter(X = pointsReconstructed_cur[0].data.cpu(),
                        win = 'FAKE_TRAIN_CURVE',
                            opts = dict(
                            title="FAKE_TRAIN_CURVE",
                            markersize=2,
                            ),
                        )
                    vis.scatter(X = points_square[0].data.cpu(),
                        win = 'REAL_TRAIN_SHEET',
                            opts = dict(
                            title = "REAL_TRAIN_SHEET",
                            markersize = 2,
                            ),
                        )
                    vis.scatter(X = pointsReconstructed_sur[0].data.cpu(),
                        win = 'FAKE_TRAIN_SHEET',
                            opts = dict(
                            title="FAKE_TRAIN_SHEET",
                            markersize=2,
                            ),
                        )
                total_time = time.time()-t0
                print('[%d: %d/%d] valid loss:  %f , %f , fetch: %f, total: %f'%(epoch, i, len(dataset_test)/opt.batchSize, loss_net.item(), validloss_accs/validloss_acc0, fetch_time, total_time))
                print('CD loss : %f ; laplacian loss : %f, CrossEntropy : %f' %((cd_total.item(), laplacian_total.item(), ce.item())))
                print('CE1 : %f, CE2 : %f, CE3 : %f, CE4 : %f' %(ce1.item(), ce2.item(), ce3.item(), ce4.item()))

    #UPDATE CURVES
    val_curve.append(val_loss.avg)
    val_curve1.append(val_loss1.avg)
    val_curve2.append(val_loss2.avg)
    val_curve3.append(val_loss3.avg)
    val_curve4.append(val_loss4.avg)
    val_curve5.append(val_loss5.avg)
    val_curve6.append(val_loss6.avg)
    val_curve7.append(val_loss7.avg)
    val_curve8.append(val_loss8.avg)

    vis.line(X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
                 Y=np.column_stack((np.array(train_curve),np.array(val_curve))),
                 win='loss_all',
                 opts=dict(title="loss_all", legend=["train_curve" + opt.env, "val_curve" + opt.env], markersize=2, ), )
    
    vis.line(X=np.column_stack((np.arange(len(train_curve1)), np.arange(len(val_curve1)))),
                 Y=np.column_stack((np.array(train_curve1),np.array(val_curve1))),
                 win='loss_ce1',
                 opts=dict(title="loss_ce1", legend=["train_curve_ref32" + opt.env, "val_curve_ref32" + opt.env], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_curve2)), np.arange(len(val_curve2)))),
                 Y=np.column_stack((np.array(train_curve2),np.array(val_curve2))),
                 win='loss_ce2',
                 opts=dict(title="loss_ce2", legend=["train_curve_ref64" + opt.env, "val_curve_ref64" + opt.env], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_curve3)), np.arange(len(val_curve3)))),
                 Y=np.column_stack((np.array(train_curve3),np.array(val_curve3))),
                 win='loss_ce3',
                 opts=dict(title="loss_ce3", legend=["train_curve_ref128" + opt.env, "val_curve_ref128" + opt.env], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_curve4)), np.arange(len(val_curve4)))),
                 Y=np.column_stack((np.array(train_curve4),np.array(val_curve4))),
                 win='loss_ce4',
                 opts=dict(title="loss_ce4", legend=["train_curve_ref256" + opt.env, "val_curve_ref256" + opt.env], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_curve5)), np.arange(len(val_curve5)))),
                 Y=np.column_stack((np.array(train_curve5),np.array(val_curve5))),
                 win='loss_cd',
                 opts=dict(title="loss_cd", legend=["train_curve_cd" + opt.env, "val_curve_cd" + opt.env], markersize=2, ), )
    vis.line(X=np.column_stack((np.arange(len(train_curve6)), np.arange(len(val_curve6)))),
                 Y=np.column_stack((np.array(train_curve6),np.array(val_curve6))),
                 win='loss_lapla',
                 opts=dict(title="loss_lapla", legend=["train_curve_lapla" + opt.env, "val_curve_lapla" + opt.env], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_curve7)), np.arange(len(val_curve7)))),
                 Y=np.column_stack((np.array(train_curve7),np.array(val_curve7))),
                 win='loss_ceall',
                 opts=dict(title="loss_ceall", legend=["train_curve_ceall" + opt.env, "val_curve_ceall" + opt.env], markersize=2, ), )
    vis.line(X=np.arange(len(val_curve8)),
                 Y=np.array(val_curve8),
                 win='real_cd',
                 opts=dict(title="real_cd", legend=["real_cd"], markersize=2, ), )

    if epoch>=opt.start_eval_epoch:
        #save mini total loss
        if best_val_loss > val_loss.avg:
            best_val_loss = val_loss.avg
            print('New best total loss : ', best_val_loss)
            print('saving net...')
            torch.save(network.state_dict(), '%s/all_epoch%d.pth' % (dir_name,epoch))
            #torch.save(network.state_dict(), '%s/all.pth' % dir_name)

        #save mini final ce loss
        if best_val_bce_final_loss > val_loss4.avg:
            best_val_bce_final_loss = val_loss4.avg
            print('New best bce final loss : ', best_val_bce_final_loss)
            print('saving net...')
            torch.save(network.state_dict(), '%s/bce_final_epoch%d.pth' % (dir_name,epoch))
            #torch.save(network.state_dict(), '%s/bce_final.pth' % dir_name)

        #save mini bce loss
        if best_val_bce_all_loss > val_loss7.avg:
            best_val_bce_all_loss = val_loss7.avg
            print('New best bce all loss : ', best_val_bce_all_loss)
            print('saving net...')
            torch.save(network.state_dict(), '%s/bce_all_epoch%d.pth' % (dir_name, epoch))
            #torch.save(network.state_dict(), '%s/bce.pth' % dir_name)

        #save mini cd loss
        if best_val_cd_loss > val_loss8.avg:
            best_val_cd_loss = val_loss8.avg
            print('New best cd loss : ', best_val_cd_loss)
            print('saving net...')
            torch.save(network.state_dict(), '%s/cd_epoch%d.pth' % (dir_name,epoch))
            #torch.save(network.state_dict(), '%s/cd.pth' % dir_name)

    torch.save(network.state_dict(), '%s/network_last.pth' % dir_name)

    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr1" : lrate1,
      "lr2" : lrate2,
      "bestval" : best_val_loss,
      "bestval_cd" : best_val_cd_loss,
      "best_val_bce_all" : best_val_bce_all_loss,
      "best_val_bce_final" : best_val_bce_final_loss
    }
    print(log_table)

    print('CD')
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item+'_cd': dataset_test.perCatValueMeter[item].avg})
    print('CE4')
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter2[item].avg)
        log_table.update({item+'_ce4': dataset_test.perCatValueMeter2[item].avg})
    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
