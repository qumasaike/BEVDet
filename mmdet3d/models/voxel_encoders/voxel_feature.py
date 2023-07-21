# The voxel_feature and bev_feature of our model.

import pdb
import math
import torch
import torch.nn as nn
import numpy as np
from ..builder import VOXEL_ENCODERS
import torch.nn.functional as F


class hourglass2d(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass2d, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes * 2,
                            inplanes * 2,
                            kernel_size=3,
                            stride=1,
                            pad=1,
                            dilation=1,
                            gn=gn)

        self.conv3 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=1,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes *
                           2) if not gn else nn.GroupNorm(32, inplanes *
                                                          2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

def convbn(in_planes,
           out_planes,
           kernel_size,
           stride,
           pad,
           dilation=1,
           gn=False,
           groups=32):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=dilation if dilation > 1 else pad,
                  dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(out_planes) if not gn else nn.GroupNorm(
            groups, out_planes))

@VOXEL_ENCODERS.register_module()
class VoxelFeature(nn.Module):
    def __init__(self, input_channels,  output_channels, Ncams, gn, scales=[1], **kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.GN = gn
        self.pools = []
        self.rpn3d_conv2s = []
        for scale in scales:
            self.pools.append(torch.nn.MaxPool3d(scale))
            self.rpn3d_conv2s.append(nn.Sequential(
                convbn(int(self.input_channels / scale), self.output_channels,
                      3, 1, 1, 1, gn=self.GN),
                nn.ReLU(inplace=True)).cuda())
        self.rpn3d_conv2s = nn.ModuleList(self.rpn3d_conv2s)


        # self.rpn3d_conv2 = nn.Sequential(
        #     convbn(self.input_channels, self.output_channels,
        #            3, 1, 1, 1, gn=self.GN),
        #     nn.ReLU(inplace=True))
        self.rpn3d_conv3 = hourglass2d(self.output_channels, gn=self.GN)

        self.init_params()
        # self.height_compress = nn.Sequential(
        #     nn.Conv2d(32*Ncams, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 16, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 8, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        #     nn.ReLU(inplace=True),
        # )
        self.height_compress = nn.Sequential(
            nn.Conv2d(32*Ncams, 32*4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32*4, 32*2, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
        )

        # self.compute_depth = nn.Sequential(
        #     nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid(),
        #     # nn.Softmax(),
        # )

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, Voxel):

        out = []
        for pool, rpn3d_conv2 in zip(self.pools, self.rpn3d_conv2s):
            B, N, C, D, H, W = Voxel.shape
            Voxel_pooled = pool(Voxel.view(B*N,*Voxel.shape[2:]))
            Voxel_pooled = Voxel_pooled.view(B, N, *Voxel_pooled.shape[1:])
            B, N, C, D, H, W = Voxel_pooled.shape

            Voxel_pooled = Voxel_pooled.view(B, N*C, D, -1)
            spatial_features = self.height_compress(Voxel_pooled)

            # depth_mask = self.compute_depth(spatial_features)
            # depth_mask = depth_mask.view(B, D, H, W)

            spatial_features = spatial_features.view(
                *spatial_features.shape[:3], H, W)
            # spatial_features = spatial_features*depth_mask.unsqueeze(1)
            
            N, C, D, H, W = spatial_features.shape

            spatial_features = spatial_features.view(N, -1, H, W)

            x = rpn3d_conv2(spatial_features)
            x = self.rpn3d_conv3(x, None, None)[0]
            out.append(x)

            # import cv2
            # show = x[0][0].detach().cpu().numpy()
            # show = (show-show.min()) / (show.max()-show.min())*255
            # cv2.imwrite('query.jpg', show)
        return out
