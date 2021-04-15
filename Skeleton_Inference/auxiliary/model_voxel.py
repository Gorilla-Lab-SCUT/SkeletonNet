from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, Conv2d, MaxPool2d, \
                     LeakyReLU, Conv3d, Tanh, Sigmoid, ReLU
import torch.nn.functional as F
import resnet

class Patch32To64(nn.Module):
    def __init__(self):
        super(Patch32To64, self).__init__()
        self.conv0 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3= nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv0_bn = nn.BatchNorm3d(32)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(256)

        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0 = nn.ConvTranspose3d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv01 = nn.ConvTranspose3d(16, 2, kernel_size=3, stride=1, padding=1)
        self.deconv00 = nn.ConvTranspose3d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_bn = nn.BatchNorm3d(64)
        self.deconv1_bn = nn.BatchNorm3d(32)
        self.deconv0_bn = nn.BatchNorm3d(16)
        self.relu = nn.LeakyReLU(negative_slope=0.02)
        self.softmax = nn.Softmax(dim=1)

    def encoder(self, x):
        x = self.relu(self.conv0_bn(self.conv0(x)))
        feat0 = x

        x = self.relu(self.conv1_bn(self.conv1(x)))
        feat1 = x

        x = self.relu(self.conv2_bn(self.conv2(x)))
        feat2 = x

        x = self.relu(self.conv3_bn(self.conv3(x)))

        return x, feat2, feat1, feat0

    def decoder(self, x, feat2, feat1, feat0):
        x = self.relu(self.deconv3_bn(self.deconv3(x)))

        x = torch.cat((feat2, x), 1)
        x = self.relu(self.deconv2_bn(self.deconv2(x)))

        x = torch.cat((feat1, x), 1)
        x = self.relu(self.deconv1_bn(self.deconv1(x)))

        x = torch.cat((feat0, x), 1)
        x = self.relu(self.deconv0_bn(self.deconv0(x)))
        x1 = self.deconv01(x)
        x2 = self.deconv00(x)
        return x1, x2

    def forward(self, x):
        x, feat2, feat1, feat0 = self.encoder(x)
        occupany1, occupany2 = self.decoder(x, feat2, feat1, feat0)
        return occupany1, occupany2

class Patch36To72(nn.Module):
    def __init__(self):
        super(Patch36To72, self).__init__()
        self.conv0 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv0_bn = nn.BatchNorm3d(32)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(256)

        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_2 = nn.ConvTranspose3d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_1 = nn.ConvTranspose3d(16, 2, kernel_size=3, stride=1, padding=1)
        self.deconv0_0 = nn.ConvTranspose3d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_bn = nn.BatchNorm3d(64)
        self.deconv1_bn = nn.BatchNorm3d(32)
        self.deconv0_2_bn = nn.BatchNorm3d(16)
        self.relu = nn.LeakyReLU(negative_slope=0.02)
        self.softmax = nn.Softmax(dim=1)

    def encoder(self, x):
        x = self.relu(self.conv0_bn(self.conv0(x)))
        feat0 = x  #18x18x18

        x = self.relu(self.conv1_bn(self.conv1(x)))
        feat1 = x #9x9x9

        x = self.relu(self.conv2_bn(self.conv2(x)))
        feat2 = x #5x5x5

        x = self.relu(self.conv3_bn(self.conv3(x))) #3x3x3
        return x,  feat2, feat1, feat0

    def decoder(self, x, feat2, feat1, feat0):
        x = self.relu(self.deconv3_bn(self.deconv3(x)))

        x = torch.cat((feat2, x), 1)
        x = self.relu(self.deconv2_bn(self.deconv2(x))) #9x9x9

        x = torch.cat((feat1, x), 1)
        x = self.relu(self.deconv1_bn(self.deconv1(x))) #18x18x18

        x = torch.cat((feat0, x), 1)
        x = self.relu(self.deconv0_2_bn(self.deconv0_2(x))) #36x36x36

        occupany1 = self.deconv0_1(x) #36x36x36
        occupany2 = self.deconv0_0(x) #72x72x72
        # occupany_softmax1 = self.softmax(occupany1)
        # occupany_softmax2 = self.softmax(occupany2)
        return occupany1, occupany2#, occupany_softmax1, occupany_softmax2

    def forward(self, local_patch):
        #local_patch: 36x36x36
        x, feat2, feat1, feat0 = self.encoder(local_patch)
        occupany1, occupany2 = self.decoder(x, feat2, feat1, feat0)
        return occupany1, occupany2

class Global_Guidance_Only64(nn.Module):
    def __init__(self):
        super(Global_Guidance_Only64, self).__init__()
        self.conv0 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3= nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv0_bn = nn.BatchNorm3d(32)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(128)

        self.deconv3 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0 = nn.ConvTranspose3d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_bn = nn.BatchNorm3d(64)
        self.deconv1_bn = nn.BatchNorm3d(32)

        self.softmax = nn.Softmax(dim=1)

    def encoder(self, x):
        x = F.relu(self.conv0_bn(self.conv0(x)))
        feat0 = x

        x = F.relu(self.conv1_bn(self.conv1(x)))
        feat1 = x

        x = F.relu(self.conv2_bn(self.conv2(x)))
        feat2 = x

        x = F.relu(self.conv3_bn(self.conv3(x)))
        return x, feat2, feat1, feat0

    def decoder(self, x, feat2, feat1, feat0):
        x = F.relu(self.deconv3_bn(self.deconv3(x)))

        x = torch.cat((feat2, x), 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))

        x = torch.cat((feat1, x), 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))

        x = torch.cat((feat0, x), 1)
        output = self.deconv0(x)

        output_softmax = self.softmax(output)
        return output, output_softmax

    def forward(self, x):
        x,  feat2, feat1, feat0 = self.encoder(x)
        output, output_softmax = self.decoder(x, feat2, feat1, feat0)
        return output, output_softmax

class Global_Structure32(nn.Module):
    def __init__(self):
        super(Global_Structure32, self).__init__()
        self.conv0 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3= nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv0_bn = nn.BatchNorm3d(32)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(128)

        self.deconv3 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_0 = nn.Conv3d(32, 2, kernel_size=3, stride=1, padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_bn = nn.BatchNorm3d(64)
        self.deconv1_bn = nn.BatchNorm3d(32)
        self.deconv0_1_bn = nn.BatchNorm3d(32)
        self.softmax = nn.Softmax(dim=1)

    def encoder(self, x):
        x = F.relu(self.conv0_bn(self.conv0(x)))
        feat0 = x

        x = F.relu(self.conv1_bn(self.conv1(x)))
        feat1 = x

        x = F.relu(self.conv2_bn(self.conv2(x)))
        feat2 = x

        x = F.relu(self.conv3_bn(self.conv3(x)))
        return x, feat2, feat1, feat0

    def decoder(self, x, feat2, feat1, feat0):
        x = F.relu(self.deconv3_bn(self.deconv3(x)))

        x = torch.cat((feat2, x), 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))

        x = torch.cat((feat1, x), 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))

        x = torch.cat((feat0, x), 1)
        x = F.relu(self.deconv0_1_bn(self.deconv0_1(x)))

        output = self.deconv0_0(x)
        return x, output

    def forward(self, x):
        x, feat2, feat1, feat0 = self.encoder(x)
        feat, output = self.decoder(x, feat2, feat1, feat0)
        return feat, output

class Global_Structure64(nn.Module):
    def __init__(self):
        super(Global_Structure64, self).__init__()
        self.conv0 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3= nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv0_bn = nn.BatchNorm3d(32)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(128)

        self.deconv3 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0 = nn.ConvTranspose3d(96, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_bn = nn.BatchNorm3d(64)
        self.deconv1_bn = nn.BatchNorm3d(32)

        self.softmax = nn.Softmax(dim=1)

    def encoder(self, x):
        x = F.relu(self.conv0_bn(self.conv0(x)))
        feat0 = x

        x = F.relu(self.conv1_bn(self.conv1(x)))
        feat1 = x

        x = F.relu(self.conv2_bn(self.conv2(x)))
        feat2 = x

        x = F.relu(self.conv3_bn(self.conv3(x)))
        return x, feat2, feat1, feat0

    def decoder(self, x, feat2, feat1, feat0, feat):
        x = F.relu(self.deconv3_bn(self.deconv3(x)))

        x = torch.cat((feat2, x), 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))

        x = torch.cat((feat1, x), 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))

        x = torch.cat((feat, feat0, x), 1)
        output = self.deconv0(x)

        output_softmax = self.softmax(output)
        return output, output_softmax

    def forward(self, x, feature):
        x,  feat2, feat1, feat0 = self.encoder(x)
        output, output_softmax = self.decoder(x, feat2, feat1, feat0, feature)
        return output, output_softmax

class Global_Guidance(nn.Module):
    def __init__(self):
        super(Global_Guidance,self).__init__()
        self.global32 = Global_Structure32()
        self.global64 = Global_Structure64()

    def forward(self, input32, input64):
        feat32, output32 = self.global32(input32)
        output64, output64_softmax = self.global64(input64, feat32)
        return output32, output64, output64_softmax

class Local_Synthesis(nn.Module):
    def __init__(self, woptfeat=False):
        super(Local_Synthesis, self).__init__()
        if woptfeat:
            self.feat0_0 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
            self.feat1_0 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        else:
            self.feat0_0 = nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1)
            self.feat1_0 = nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1)
        self.feat0_1 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.feat1_1 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.feat0_0_bn = nn.BatchNorm3d(16)
        self.feat0_1_bn = nn.BatchNorm3d(32)
        self.feat1_0_bn = nn.BatchNorm3d(16)
        self.feat1_1_bn = nn.BatchNorm3d(32)

        if woptfeat:
            self.conv0 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        else:
            self.conv0 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv0_bn = nn.BatchNorm3d(32)
        self.conv1_bn = nn.BatchNorm3d(64)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.conv3_bn = nn.BatchNorm3d(128)

        self.deconv3 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 64, kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose3d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_2 = nn.ConvTranspose3d(96, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv0_1 = nn.ConvTranspose3d(16, 2, kernel_size=3, stride=1, padding=1)
        self.deconv0_0 = nn.ConvTranspose3d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.deconv3_bn = nn.BatchNorm3d(128)
        self.deconv2_bn = nn.BatchNorm3d(64)
        self.deconv1_bn = nn.BatchNorm3d(32)
        self.deconv0_2_bn = nn.BatchNorm3d(16)
        self.softmax = nn.Softmax(dim=1)

    def encoder(self, x, feat):
        x = F.relu(self.conv0_bn(self.conv0(x)))
        feat0 = x  #18x18x18

        x = torch.cat((x, feat), 1)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        feat1 = x #9x9x9

        x = F.relu(self.conv2_bn(self.conv2(x)))
        feat2 = x #5x5x5

        x = F.relu(self.conv3_bn(self.conv3(x))) 
        return x,  feat2, feat1, feat0

    def decoder(self, x, feat2, feat1, feat0, feat):
        x = F.relu(self.deconv3_bn(self.deconv3(x)))

        x = torch.cat((feat2, x), 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x))) #9x9x9

        x = torch.cat((feat1, x), 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x))) #18x18x18

        x = torch.cat((feat, feat0, x), 1)
        x = F.relu(self.deconv0_2_bn(self.deconv0_2(x))) #36x36x36

        occupany1 = self.deconv0_1(x) #36x36x36
        occupany2 = self.deconv0_0(x) #72x72x72
        # occupany_softmax1 = self.softmax(occupany1)
        # occupany_softmax2 = self.softmax(occupany2)
        return occupany1, occupany2#, occupany_softmax1, occupany_softmax2

    def forward(self, global_patch, local_patch):
        #global_patch: 18x18x18 local_patch: 36x36x36
        feat_conv0 = F.relu(self.feat0_0_bn(self.feat0_0(global_patch)))
        feat_conv0 = F.relu(self.feat0_1_bn(self.feat0_1(feat_conv0)))
        feat_conv1 = F.relu(self.feat1_0_bn(self.feat1_0(global_patch)))
        feat_conv1 = F.relu(self.feat1_1_bn(self.feat1_1(feat_conv1)))
        x, feat2, feat1, feat0 = self.encoder(local_patch, feat_conv0)
        occupany1, occupany2 = self.decoder(x, feat2, feat1, feat0, feat_conv1)
        return occupany1, occupany2


########R2N2 --binvox 32
class Embedding(nn.Module):
    def __init__(self,bottleneck_size):
        super(Embedding, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.n_gru_vox = 2
        self.gf_dim = 128
        self.fc = nn.Linear(self.bottleneck_size, self.gf_dim*self.n_gru_vox*self.n_gru_vox*self.n_gru_vox)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1,self.gf_dim, self.n_gru_vox,self.n_gru_vox,self.n_gru_vox)
        return x

from torch.nn import Linear, Conv2d, MaxPool2d, \
                     LeakyReLU, Conv3d, ConvTranspose3d, Tanh, Sigmoid, ReLU
class VoxelDecoder(nn.Module):
    def __init__(self):
        print("\ninitializing \"decoder\"")
        super(VoxelDecoder, self).__init__()
        self.n_deconvfilter = [128, 128, 128, 64, 32, 2]

        #3d conv1
        conv1_kernel_size = 3
        self.conv1 = ConvTranspose3d(in_channels= self.n_deconvfilter[0], \
                            out_channels= self.n_deconvfilter[1], \
                            kernel_size= conv1_kernel_size,  stride=2, \
                            padding = int((conv1_kernel_size - 1) / 2), \
                            output_padding = int((conv1_kernel_size - 1) / 2))

        #3d conv2
        conv2_kernel_size = 3
        self.conv2 = ConvTranspose3d(in_channels= self.n_deconvfilter[1], \
                            out_channels= self.n_deconvfilter[2], \
                            kernel_size= conv1_kernel_size,  stride=2, \
                            padding = int((conv1_kernel_size - 1) / 2), \
                            output_padding = int((conv1_kernel_size - 1) / 2))

        #3d conv3
        conv3_kernel_size = 3
        self.conv3 = ConvTranspose3d(in_channels= self.n_deconvfilter[2], \
                            out_channels= self.n_deconvfilter[3], \
                            kernel_size= conv1_kernel_size,  stride=2, \
                            padding = int((conv1_kernel_size - 1) / 2), \
                            output_padding = int((conv1_kernel_size - 1) / 2))

        #3d conv4
        conv4_kernel_size = 3
        self.conv4 = ConvTranspose3d(in_channels= self.n_deconvfilter[3], \
                            out_channels= self.n_deconvfilter[4], \
                            kernel_size= conv1_kernel_size,  stride=2, \
                            padding = int((conv1_kernel_size - 1) / 2), \
                            output_padding = int((conv1_kernel_size - 1) / 2))

        #3d conv5
        conv5_kernel_size = 3
        self.conv5 = Conv3d(in_channels= self.n_deconvfilter[4], \
                            out_channels= self.n_deconvfilter[5], \
                            kernel_size= conv5_kernel_size, \
                            padding = int((conv5_kernel_size - 1) / 2))

        #nonlinearities of the network
        self.leaky_relu = LeakyReLU(negative_slope= 0.01)

        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(32)
        self.bn5 = nn.BatchNorm3d(2)

    def forward(self, x):
        out1 = self.leaky_relu(self.bn1(self.conv1(x)))
        out2 = self.leaky_relu(self.bn2(self.conv2(out1)))
        out3 = self.leaky_relu(self.bn3(self.conv3(out2)))
        out4 = self.leaky_relu(self.bn4(self.conv4(out3)))
        out5_1 = self.conv5(out4)
        out5_2 = self.softmax(out5_1)
        return out5_1, out5_2

import resnet
class SVR_R2N2(nn.Module):
    def __init__(self, voxel_size = 32, bottleneck_size = 1024, pretrained_encoder=False): #1024
        super(SVR_R2N2, self).__init__()
        self.voxel_size = voxel_size
        self.bottleneck_size = bottleneck_size
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=self.bottleneck_size)
        self.embedding = Embedding( bottleneck_size = self.bottleneck_size)
        self.decoder = VoxelDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.embedding(x)
        output, voxelreconstruct = self.decoder(x)
        return output, voxelreconstruct

if __name__ == '__main__':
    input1 = Variable(torch.randn(1, 16, 32, 32, 32))
    input2 = Variable(torch.randn(1, 16, 64, 64, 64))
    model = Global_Guidance()
    model = model.cuda()
    occupany1, occupany2, _ = model(input1.cuda(), input2.cuda())
    print('occupany1, occupany2:', occupany1.size(), occupany2.size())

    global_patch = Variable(torch.randn(1, 2, 18, 18, 18))
    local_patch = Variable(torch.randn(1, 16, 36, 36, 36))
    model = Local_Synthesis()
    model = model.cuda()
    occupany = model(global_patch.cuda(), local_patch.cuda())
    print('occupany :', occupany.size())