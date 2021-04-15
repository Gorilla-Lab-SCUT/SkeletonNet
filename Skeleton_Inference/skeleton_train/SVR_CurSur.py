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
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--model_ae', type=str, default = '',  help='model path')
parser.add_argument('--num_points_line', type=int, default = 600,  help='number of curve points')
parser.add_argument('--nb_primitives_line', type=int, default = 20,  help='number of primitives')
parser.add_argument('--num_points_square', type=int, default = 2000,  help='number of sheet points')
parser.add_argument('--nb_primitives_square', type=int, default = 20,  help='number of primitives')
parser.add_argument('--bottleneck_size', type=int, default=512, help='bottleneck_size')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="SVR_CurSur"   ,  help='visdom env')
parser.add_argument('--fix_decoder',  default = 'False'   ,  help='if set to True, on the pointNet encoder is trained')
parser.add_argument('--k1',type=float,default=0.2)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--lrDecay', type=float, default=0.1)
parser.add_argument('--lrStep', type=float, default=300)
parser.add_argument('--start_eval_epoch', type=float, default=100)
parser.add_argument('--category', type=str, default='all/chair')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--white_bg', action='store_true')
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

# Create train/test dataloader on new views and test dataset on new models
if opt.category == "all":
    category_select = None
else:
    category_select = (opt.category).split('_')
dataset = ShapeNet(train=True, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points, rotate=opt.rotate, white_bg=opt.white_bg)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, drop_last=True,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(train=False, class_choice=category_select, npoints_line=opt.super_points, npoints_square=opt.super_points, npoints_skeleton=opt.super_points, rotate=opt.rotate, white_bg=opt.white_bg)
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
    print('Previous ae weight loaded!')
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
# print(network)

lrate = opt.lr
params_dict = dict(network.named_parameters())
params = []
if opt.fix_decoder=='True':
    optimizer = optim.Adam(network.encoder.parameters(), lr = lrate)
else:
    optimizer = optim.Adam(network.parameters(), lr = lrate)

num_batch = len(dataset) / opt.batchSize
train_loss = AverageValueMeter()
loss1 = AverageValueMeter()
loss2 = AverageValueMeter()
val_loss = AverageValueMeter()
val_loss1 = AverageValueMeter()
val_loss2 = AverageValueMeter()
val_loss3 = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')

train_curve = []
train_curve1 = []
train_curve2 = []
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
    network.train()
    if (epoch+1)%opt.lrStep == 0:
        if opt.fix_decoder=='True':
            optimizer = optim.Adam(network.encoder.parameters(), lr = lrate * opt.lrDecay)
        else:
            optimizer = optim.Adam(network.parameters(), lr = lrate * opt.lrDecay)
        lrate = lrate * opt.lrDecay
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        if opt.rotate:
            img, rotation, points_skeleton, points_line, points_square, cat, mod = data
            rotation = rotation.cuda()
        else:
            img, points_skeleton, points_line, points_square, cat, mod = data
        img = img.cuda()
        points_skeleton = points_skeleton.cuda()
        points_line = points_line.cuda()
        points_square = points_square.cuda()

        pointsReconstructed_cur, pointsReconstructed_sur = network.forward_inference(img, grid1, grid2)
        cur_dist1, cur_dist2 = distChamfer(points_line, pointsReconstructed_cur)
        cur_laplacian_smooth = curve_laplacian(pointsReconstructed_cur, opt.nb_primitives_line, lines_adjacent_tensor)
        sur_dist1, sur_dist2 = distChamfer(points_square, pointsReconstructed_sur)
        sur_laplacian_smooth = surface_laplacian(pointsReconstructed_sur, opt.nb_primitives_square, vertex_adj_matrix_tensor)

        cd_total = torch.mean(cur_dist1) + torch.mean(cur_dist2) + torch.mean(sur_dist1) + torch.mean(sur_dist2)
        laplacian_total = torch.mean(cur_laplacian_smooth) + torch.mean(sur_laplacian_smooth)
        loss_net = cd_total + opt.k1* laplacian_total
        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        loss1.update(cd_total.item())
        loss2.update(laplacian_total.item())

        optimizer.step()
        # VIZUALIZE
        if i%10 <= 0:
            vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE TRAIN', opts=dict(title="INPUT IMAGE TRAIN"))
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
        print('[%d: %d/%d] train loss:  %f , %f ' %(epoch, i, len(dataset)/opt.batchSize, loss_net.item(), trainloss_accs/trainloss_acc0))
        print('CD loss : %f ; laplacian loss : %f' %((cd_total.item(), laplacian_total.item())))
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

    if epoch>=opt.start_eval_epoch:
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

                pointsReconstructed_cur, pointsReconstructed_sur = network.forward_inference(img, grid1, grid2)
                cur_dist1, cur_dist2 = distChamfer(points_line, pointsReconstructed_cur)
                cur_laplacian_smooth = curve_laplacian(pointsReconstructed_cur, opt.nb_primitives_line, lines_adjacent_tensor)
                sur_dist1, sur_dist2 = distChamfer(points_square, pointsReconstructed_sur)
                sur_laplacian_smooth = surface_laplacian(pointsReconstructed_sur, opt.nb_primitives_square, vertex_adj_matrix_tensor)
                pointsReconstructed = torch.cat((pointsReconstructed_cur, pointsReconstructed_sur), dim=1)
                dist1, dist2 = distChamfer(points_skeleton, pointsReconstructed)
                real_cd = torch.mean(dist1) + torch.mean(dist2)

                cd_total = torch.mean(cur_dist1) + torch.mean(cur_dist2) + torch.mean(sur_dist1) + torch.mean(sur_dist2)
                laplacian_total = torch.mean(cur_laplacian_smooth) + torch.mean(sur_laplacian_smooth)
                loss_net = cd_total + opt.k1* laplacian_total
                validloss_accs = validloss_accs * 0.99 + loss_net.item()
                validloss_acc0 = validloss_acc0 * 0.99 + 1 
                val_loss.update(loss_net.item())
                val_loss1.update(cd_total.item())
                val_loss2.update(laplacian_total.item())
                val_loss3.update(real_cd.item())
                dataset_test.perCatValueMeter[cat[0]].update(real_cd.item())
                if i%5 ==0 :
                    vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE VAL', opts=dict(title="INPUT IMAGE VAL"))
                    vis.scatter(X = points_line[0].data.cpu(),
                        win = 'REAL_VAL_CURVE',
                            opts = dict(
                            title = "REAL_VAL_CURVE",
                            markersize = 2,
                            ),
                        )
                    vis.scatter(X = pointsReconstructed_cur[0].data.cpu(),
                        win = 'FAKE_VAL_CURVE',
                            opts = dict(
                            title="FAKE_VAL_CURVE",
                            markersize=2,
                            ),
                        )
                    vis.scatter(X = points_square[0].data.cpu(),
                        win = 'REAL_VAL_SHEET',
                            opts = dict(
                            title = "REAL_VAL_SHEET",
                            markersize = 2,
                            ),
                        )
                    vis.scatter(X = pointsReconstructed_sur[0].data.cpu(),
                        win = 'FAKE_VAL_SHEET',
                            opts = dict(
                            title="FAKE_VAL_SHEET",
                            markersize=2,
                            ),
                        )
                print('[%d: %d/%d] valid loss:  %f , %f ' %(epoch, i, len(dataset_test)/opt.batchSize, loss_net.item(), validloss_accs/validloss_acc0))
                print('CD loss : %f ; laplacian loss : %f' %((cd_total.item(),laplacian_total.item())))

    # UPDATE CURVES
    val_curve.append(val_loss.avg)
    val_curve1.append(val_loss1.avg)
    val_curve2.append(val_loss2.avg)
    val_curve3.append(val_loss3.avg)

    #UPDATE CURVES
    vis.line(
        X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
        Y=np.log(np.column_stack((np.array(train_curve), np.array(val_curve)))),
        win='All loss log',
        opts=dict(title="All loss", legend=["train_curve" + opt.env, "val_curve" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve1)), np.arange(len(val_curve1)))),
        Y=np.log(np.column_stack((np.array(train_curve1), np.array(val_curve1)))),
        win='CD loss log',
        opts=dict(title="CD loss", legend=["train_curve1" + opt.env, "val_curve1" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve2)), np.arange(len(val_curve2)))),
        Y=np.log(np.column_stack((np.array(train_curve2), np.array(val_curve2)))),
        win='Laplacian loss log',
        opts=dict(title="Laplacian loss", legend=["train_curve2" + opt.env, "val_curve2" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.arange(len(val_curve3)),
        Y=np.log(np.array(val_curve3)),
        win='Real CD loss log',
        opts=dict(title="Real CD loss", legend=["val_curve3" + opt.env],
                  markersize=2, ), )

    if epoch>=opt.start_eval_epoch:
        if best_val_loss > val_loss3.avg:
            best_val_loss = val_loss3.avg
            print('New best loss : ', best_val_loss)
            print('saving net...')
            torch.save(network.state_dict(), '%s/network_epoch%d.pth' % (dir_name,epoch))
    torch.save(network.state_dict(), '%s/network.pth' % dir_name)
    torch.save(network.state_dict(), '%s/network_last.pth' % dir_name)


    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "super_points" : opt.super_points,
      "bestval" : best_val_loss,
    }
    print(log_table)

    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})

    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
