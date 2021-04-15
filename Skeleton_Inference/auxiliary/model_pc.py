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
#import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
import resnet

#UTILITIES
class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
        self.trans = trans

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

#  Square-only fitting 
class AE_SurSkeNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 512, nb_primitives = 1):
        super(AE_SurSkeNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans = False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
            )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives)])

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points/self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

class SVR_SurSkeNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 512, nb_primitives = 5, pretrained_encoder = False, cuda=True):
        super(SVR_SurSkeNet, self).__init__()
        self.usecuda = cuda
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=self.bottleneck_size)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points/self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.usecuda:
                rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            else:
                rand_grid = Variable(torch.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

#  Line-only fitting 
class AE_CurSkeNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 512, nb_primitives = 1):
        super(AE_CurSkeNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans = False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
            )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 1 +self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 1, self.num_points/self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            rand_zero = Variable(torch.zeros(x.size(0), 1, self.num_points / self.nb_primitives).cuda())
            rand_grid = torch.cat((rand_grid, rand_zero), 1).contiguous()

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
        outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

class SVR_CurSkeNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 512, nb_primitives = 5, pretrained_encoder = False, cuda=True):
        super(SVR_CurSkeNet, self).__init__()
        self.usecuda = cuda
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=self.bottleneck_size)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 1 +self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 1, self.num_points/self.nb_primitives))
            rand_grid.data.uniform_(0, 1)

            rand_zero = Variable(torch.zeros(x.size(0), 1, self.num_points / self.nb_primitives).cuda())
            rand_grid = torch.cat((rand_grid, rand_zero), 1).contiguous()

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.usecuda:
                rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            else:
                rand_grid = Variable(torch.FloatTensor(grid[i]))

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

# Line Square joint fitting, compute loss after concat
class SVR_CurSur_Joint(nn.Module):
    def __init__(self, pretrained_encoder=False, num_points_line = 2048, num_points_square = 2048, bottleneck_size = 512, nb_primitives_line = 1, nb_primitives_square = 1):
        super(SVR_CurSur_Joint, self).__init__()
        self.num_points_line = num_points_line
        self.num_points_square = num_points_square
        self.bottleneck_size = bottleneck_size
        self.nb_primitives_line = nb_primitives_line
        self.nb_primitives_square = nb_primitives_square
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=self.bottleneck_size)
        self.decoder_line = nn.ModuleList([PointGenCon(bottleneck_size = 1 +self.bottleneck_size) for i in range(0,self.nb_primitives_line)])
        self.decoder_square = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives_square)])

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs_1 = []
        x1 = x
        for i in range(0,self.nb_primitives_line):
            rand_grid = Variable(torch.cuda.FloatTensor(x1.size(0), 1, self.num_points_line/self.nb_primitives_line))
            rand_grid.data.uniform_(0,1)
            y = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs_1.append(self.decoder_line[i](y))

        outs_2 = []
        x2 = x
        for i in range(0, self.nb_primitives_square):
            rand_grid = Variable(torch.cuda.FloatTensor(x2.size(0),2, self.num_points_square/self.nb_primitives_square))
            rand_grid.data.uniform_(0,1)
            y = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs_2.append(self.decoder_square[i](y))
        return torch.cat(outs_1,2).contiguous().transpose(2,1).contiguous(), torch.cat(outs_2,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid1, grid2):
        x = self.encoder(x)
        x1 = x
        outs_1 = []
        for i in range(0,self.nb_primitives_line):
            grid = torch.tensor(grid1[i], dtype=torch.float32)
            grid = grid.transpose(0,1).contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x1.size(0),grid.size(1), grid.size(2)).contiguous())
            grid = grid.cuda()
            y = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), grid.size(2)).contiguous()
            y = torch.cat((grid, y), 1).contiguous()
            outs_1.append(self.decoder_line[i](y))

        x2 = x
        outs_2 = []
        for i in range(0, self.nb_primitives_square):
            grid = torch.tensor(grid2[i], dtype=torch.float32).cuda()
            grid = grid.transpose(0,1).contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x2.size(0),grid.size(1), grid.size(2)).contiguous())
            grid = grid.cuda()
            y = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), grid.size(2)).contiguous()
            y = torch.cat((grid, y), 1).contiguous()
            outs_2.append(self.decoder_square[i](y))
        return torch.cat(outs_1,2).contiguous().transpose(2,1).contiguous(), torch.cat(outs_2,2).contiguous().transpose(2,1).contiguous()

# Line Square sperate fitting, compute loss individually
class AE_CurSur(nn.Module):
    def __init__(self, num_points =2500, num_points_line = 2048, num_points_square = 2048, bottleneck_size = 1024, nb_primitives_line = 1, nb_primitives_square = 1):
        super(AE_CurSur, self).__init__()
        self.num_points = num_points
        self.num_points_line = num_points_line
        self.num_points_square = num_points_square
        self.bottleneck_size = bottleneck_size
        self.nb_primitives_line = nb_primitives_line
        self.nb_primitives_square = nb_primitives_square
        self.encoder_line = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans = False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
            )
        self.encoder_square = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans = False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
            )
        self.decoder_line = nn.ModuleList([PointGenCon(bottleneck_size = 1 +self.bottleneck_size) for i in range(0,self.nb_primitives_line)])
        self.decoder_square = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives_square)])

    def forward(self, x1, x2):
        x1 = self.encoder_line(x1)
        outs_1 = []
        for i in range(0, self.nb_primitives_line):
            rand_grid = Variable(torch.cuda.FloatTensor(x1.size(0), 1, self.num_points_line/self.nb_primitives_line))
            rand_grid.data.uniform_(0, 1)
            rand_zero = Variable(torch.zeros(x1.size(0), 1, self.num_points_line / self.nb_primitives_line).cuda())
            rand_grid = torch.cat((rand_grid, rand_zero), 1).contiguous()

            y = x1.unsqueeze(2).expand(x1.size(0), x1.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
        outs_1.append(self.decoder_line[i](y))

        x2 = self.encoder_square(x2)
        outs_2 = []
        for i in range(0,self.nb_primitives_square):
            rand_grid = Variable(torch.cuda.FloatTensor(x2.size(0), 2, self.num_points_square/self.nb_primitives_square))
            rand_grid.data.uniform_(0,1)
            y = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs_2.append(self.decoder_square[i](y))
        return torch.cat(outs_1, 2).contiguous().transpose(2,1).contiguous(), torch.cat(outs_2,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x1, x2, grid1, grid2):
        x1 = self.encoder_line(x1)
        outs_1 = []
        for i in range(0, self.nb_primitives_line):
            rand_grid = Variable(torch.cuda.FloatTensor(grid1[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x1.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x1.unsqueeze(2).expand(x1.size(0), x1.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs_1.append(self.decoder_line[i](y))

        x2 = self.encoder_square(x2)
        outs_2 = []
        for i in range(0,self.nb_primitives_square):
            rand_grid = Variable(torch.cuda.FloatTensor(grid2[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x2.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs_2.append(self.decoder_square[i](y))
        return torch.cat(outs_1, 2).contiguous().transpose(2,1).contiguous(), torch.cat(outs_2,2).contiguous().transpose(2,1).contiguous()

class ResNetfeat(nn.Module):
    def __init__(self, pretrained_encoder=False, bottleneck_size = 512):
        super(ResNetfeat, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.bottleneck_size = bottleneck_size
        self.extractor = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.fc1 = nn.Sequential(nn.Linear(1024, self.bottleneck_size),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, self.bottleneck_size),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU())

    def forward(self, x):
        x = self.extractor(x)
        vector1 = self.fc1(x)
        vector2 = self.fc2(x)
        return vector1, vector2

class SVR_CurSur(nn.Module):
    def __init__(self, pretrained_encoder=False, num_points_line = 2048, num_points_square = 2048, bottleneck_size = 512, nb_primitives_line = 1, nb_primitives_square = 1):
        super(SVR_CurSur, self).__init__()
        self.num_points_line = num_points_line
        self.num_points_square = num_points_square
        self.bottleneck_size = bottleneck_size
        self.nb_primitives_line = nb_primitives_line
        self.nb_primitives_square = nb_primitives_square
        self.pretrained_encoder = pretrained_encoder
        self.encoder = ResNetfeat(pretrained_encoder=False, bottleneck_size = self.bottleneck_size)
        self.decoder_line = nn.ModuleList([PointGenCon(bottleneck_size = 1 +self.bottleneck_size) for i in range(0,self.nb_primitives_line)])
        self.decoder_square = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives_square)])

    def forward(self, x):
        x1, x2 = self.encoder(x)
        outs_1 = []
        for i in range(0,self.nb_primitives_line):
            rand_grid = Variable(torch.cuda.FloatTensor(x1.size(0), 1, self.num_points_line/self.nb_primitives_line))
            rand_grid.data.uniform_(0,1)
            y = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs_1.append(self.decoder_line[i](y))

        outs_2 = []
        for i in range(0, self.nb_primitives_square):
            rand_grid = Variable(torch.cuda.FloatTensor(x2.size(0),2, self.num_points_square/self.nb_primitives_square))
            rand_grid.data.uniform_(0,1)
            y = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs_2.append(self.decoder_square[i](y))
        return torch.cat(outs_1,2).contiguous().transpose(2,1).contiguous(), torch.cat(outs_2,2).contiguous().transpose(2,1).contiguous()

    #pytorch 1.0.1
    def forward_inference(self, x, grid1, grid2):
        x1, x2 = self.encoder(x)
        outs_1 = []
        for i in range(0,self.nb_primitives_line):
            grid = torch.tensor(grid1[i], dtype=torch.float32)
            grid = grid.transpose(0,1).contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x1.size(0),grid.size(1), grid.size(2)).contiguous())
            grid = grid.cuda()
            y = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), grid.size(2)).contiguous()
            y = torch.cat((grid, y), 1).contiguous()
            outs_1.append(self.decoder_line[i](y))

        outs_2 = []
        for i in range(0, self.nb_primitives_square):
            grid = torch.tensor(grid2[i], dtype=torch.float32).cuda()
            grid = grid.transpose(0,1).contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x2.size(0),grid.size(1), grid.size(2)).contiguous())
            grid = grid.cuda()
            y = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), grid.size(2)).contiguous()
            y = torch.cat((grid, y), 1).contiguous()
            outs_2.append(self.decoder_square[i](y))
        return torch.cat(outs_1,2).contiguous().transpose(2,1).contiguous(), torch.cat(outs_2,2).contiguous().transpose(2,1).contiguous()

if __name__ == '__main__':
    print('testing SurfaceToSkeleton...')
    sim_data = Variable(torch.rand(32, 3, 2500))
    model = SurfaceToSkeleton()
    model.cuda()
    out1, out2 = model(sim_data.cuda())
    #out = model(sim_data)
    print(out1.size(), out2.size())

    print('testing ImgToSkeleton...')
    sim_data = Variable(torch.rand(32, 3, 224, 224))
    model = ImgToSkeleton()
    model.cuda()
    out1, out2 = model(sim_data.cuda())
    #out = model(sim_data)
    print(out1.size(), out2.size())