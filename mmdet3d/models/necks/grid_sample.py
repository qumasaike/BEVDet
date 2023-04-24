# The gridsample of our model.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tools.point_sample import bilinear_grid_sample_test
import time
from ..builder import NECKS
import numpy as np

def project_pseudo_lidar_to_rectcam(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([-ys, -zs, xs], dim=-1)


def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n, 1), device=pts_3d_rect.device)
    # pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3
    # pts_2d[:, 0] /= pts_2d[:, 2]
    # pts_2d[:, 1] /= pts_2d[:, 2] # changed by me
    # pdb.set_trace()
    kept_not = pts_2d[..., -1] < 0
    kept_not = kept_not.view(-1, 1).expand(-1, 2)
    pts_2d = torch.stack(
        (pts_2d[:, 0]/pts_2d[:, 2], pts_2d[:, 1]/pts_2d[:, 2]), dim=1)
    # pts_2d[pts_2d[:,2]<0] = torch.tensor([-1000.,-1000.,1]).cuda()
    pts_2d = torch.where(kept_not, torch.tensor(-1000.,
                         device=pts_2d.device), pts_2d.clone().detach())
    pts_2d = torch.where(torch.abs(pts_2d) > 10000, torch.tensor(-1000.,
                         device=pts_2d.device), pts_2d.clone().detach())
    return pts_2d[:, 0:2]

@NECKS.register_module()
class GridSample(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = np.array(voxel_size)

        self.grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(self.grid_size).astype(np.int64)
        self.prepare_coordinates_3d(point_cloud_range, voxel_size)
        self.init_params()

    def prepare_coordinates_3d(self, point_cloud_range, voxel_size,  sample_rate=(1, 1, 1)):
        self.X_MIN, self.Y_MIN, self.Z_MIN = point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = voxel_size
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = self.grid_size.tolist()

        self.CV_DEPTH_MIN = point_cloud_range[0]
        self.CV_DEPTH_MAX = point_cloud_range[3]

        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()

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

    def forward(self, inputs):

        # (feature, rots, trans, intrins, post_rots, post_trans, bda) = x
        # rots 和 trans 是外参， intrincs 是内参
        # post_rots, post_trans 是数据增强在图像平面做的平移和旋转
        # bda是bev data augumentaion,在bev平面上做了一些变换。
        (features, rots, trans, intrins, post_rots, post_trans, bda, (img_H, img_W)) = inputs
        B, N, C, feature_H, feature_W = features.shape
        coordinates_3d = self.coordinates_3d.cuda()
        coordinates_3d = coordinates_3d.expand((B,*coordinates_3d.shape))
        B_, bev_D, bev_H, bev_W, _ = coordinates_3d.shape

        #体素坐标转为各个相机的相机坐标系
        c3d = bda.view(B, 1, 1, 1, 3, 3).matmul(coordinates_3d.unsqueeze(-1)).squeeze(-1)
        c3d = c3d - trans.view(B, N, 1, 1, 1, 3)
        c3d = torch.inverse(rots).view(B, N, 1, 1, 1, 3, 3).matmul(c3d.unsqueeze(-1)).squeeze(-1)

        # 转为像素uv坐标系
        c3d_image = intrins.view(B, N, 1, 1, 1, 3, 3).matmul(c3d.unsqueeze(-1)).squeeze(-1)
        coord_img = torch.stack(
            (c3d_image[..., 0]/c3d_image[..., 2], c3d_image[..., 1]/c3d_image[..., 2]), dim=-1)
        kept_not = coord_img[..., -1] < 0
        kept_not = kept_not.unsqueeze(-1).expand((B, N, bev_D, bev_H, bev_W, 2))
        coord_img = torch.where(kept_not, torch.tensor(-10000.,
                            device=coord_img.device), coord_img.clone().detach())
        coord_img = torch.where(torch.abs(coord_img) > 10000, torch.tensor(-1000.,
                            device=coord_img.device), coord_img.clone().detach())
        
        # 像素uv坐标系转换到图像数据增强后的uv坐标系
        coord_img = post_rots[:,:,:2,:2].view(B, N, 1, 1, 1, 2, 2) \
                    .matmul(coord_img.unsqueeze(-1)).squeeze(-1) \
                    + post_trans[:,:,:2].view(B, N, 1, 1, 1, 2)

        crop_x1, crop_x2 = 0, img_W
        crop_y1, crop_y2 = 0, img_H
        norm_coord_img = \
            (coord_img - torch.as_tensor([crop_x1, crop_y1], device=coord_img.device)) /\
            torch.as_tensor([crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1],device=coord_img.device)
        norm_coord_img = norm_coord_img * 2. - 1.

        Voxels = F.grid_sample(
                              features.view(B*N,C,feature_H,feature_W), 
                              norm_coord_img.view(B*N, bev_D*bev_H, bev_W,2), align_corners=True).view(B, N, C, bev_D, bev_H, bev_W)
        
        return Voxels
