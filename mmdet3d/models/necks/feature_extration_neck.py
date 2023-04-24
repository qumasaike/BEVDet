import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from mmdet3d.models.voxel_encoders.voxel_feature import convbn
from ..builder import NECKS



class upconv_module(nn.Module):
    def __init__(self, in_channels, up_channels):
        super(upconv_module, self).__init__()
        self.num_stage = len(in_channels) - 1
        self.conv = nn.ModuleList()
        self.redir = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            self.conv.append(
                convbn(in_channels[0] if stage_idx == 0 else up_channels[stage_idx - 1], up_channels[stage_idx], 3, 1, 1, 1)
            )
            self.redir.append(
                convbn(in_channels[stage_idx + 1], up_channels[stage_idx], 3, 1, 1, 1)
            )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, feats):
        x = feats[0].contiguous()
        for stage_idx in range(self.num_stage):
            x = self.conv[stage_idx](x).contiguous()
            redir = self.redir[stage_idx](feats[stage_idx + 1]).contiguous()
            x = F.relu(self.up(x) + redir).contiguous()
        return x

@NECKS.register_module()
class FeatureExtractionNeck(nn.Module):
    def __init__(self, cfg):
        super(FeatureExtractionNeck, self).__init__()

        self.cfg = cfg
        self.in_dims = cfg.in_dims
        self.with_upconv = cfg.with_upconv
        self.start_level = cfg.start_level
        self.cat_img_feature = cfg.cat_img_feature

        self.sem_dim = cfg.sem_dim
        self.stereo_dim = cfg.stereo_dim
        self.spp_dim = getattr(cfg, 'spp_dim', 32)

        self.spp_branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(s, stride=s),
                convbn(self.in_dims[-1],
                       self.spp_dim,
                       1, 1, 0,
                       gn=cfg.GN,
                       groups=min(32, self.spp_dim)),
                nn.ReLU(inplace=True))
            for s in [(64, 64), (32, 32), (16, 16), (8, 8)]])

        concat_dim = self.spp_dim * len(self.spp_branches) + sum(self.in_dims[self.start_level:])

        if self.with_upconv:
            assert self.start_level == 2
            self.upconv_module = upconv_module([concat_dim, self.in_dims[1], self.in_dims[0]], [64, 32])
            stereo_dim = 32
        else:
            stereo_dim = concat_dim
            assert self.start_level >= 1

        self.lastconv = nn.Sequential(
            convbn(stereo_dim, self.stereo_dim[0], 3, 1, 1, gn=cfg.GN),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stereo_dim[0], self.stereo_dim[1],
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=False))

        if self.cat_img_feature:
            self.rpnconv = nn.Sequential(
                convbn(concat_dim, self.sem_dim[0], 3, 1, 1, 1, gn=cfg.GN),
                nn.ReLU(inplace=True),
                convbn(self.sem_dim[0], self.sem_dim[1], 3, 1, 1, gn=cfg.GN),
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):
        feat_shape = tuple(feats[self.start_level].shape[2:])
        assert len(feats) == len(self.in_dims)

        spp_branches = []
        for branch_module in self.spp_branches:
            x = branch_module(feats[-1])
            x = F.interpolate(
                x, feat_shape,
                mode='bilinear',
                align_corners=True)
            spp_branches.append(x)

        concat_feature = torch.cat((*feats[self.start_level:], *spp_branches), 1)
        stereo_feature = concat_feature

        if self.with_upconv:
            stereo_feature = self.upconv_module([stereo_feature, feats[1], feats[0]])

        stereo_feature = self.lastconv(stereo_feature)

        if self.cat_img_feature:
            sem_feature = self.rpnconv(concat_feature)
        else:
            sem_feature = None

        return stereo_feature, sem_feature
