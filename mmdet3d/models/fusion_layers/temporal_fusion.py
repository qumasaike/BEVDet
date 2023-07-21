# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import  build_norm_layer

from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from mmdet.models.utils import LearnedPositionalEncoding
import numpy as np
from ..builder import FUSION_LAYERS



ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def unitQ2RotateM_L(unitQ):
    [w, x, y, z] = unitQ
    rotateM = np.zeros((3, 3))
    rotateM[0, 0] = 1 - 2*y**2 - 2*z**2
    rotateM[0, 1] = 2*x*y + 2*w*z
    rotateM[0, 2] = 2*x*z - 2*w*y
    rotateM[1, 0] = 2*x*y - 2*w*z
    rotateM[1, 1] = 1 - 2*x**2 - 2*z**2
    rotateM[1, 2] = 2*y*z + 2*w*x
    rotateM[2, 0] = 2*x*z + 2*w*y
    rotateM[2, 1] = 2*y*z - 2*w*x
    rotateM[2, 2] = 1 - 2*x**2 - 2*y**2
    return rotateM.T


@FUSION_LAYERS.register_module()
class TemporalSelfAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 voxel_size=[0.2, 0.2, 0.2],
                 point_cloud_range=[],
                 embed_dims=64,
                 num_heads=1,
                 num_levels=1,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(self.grid_size).astype(np.int64)

        self.bev_w, self.bev_h = self.grid_size[0], self.grid_size[1]
        self.init_weights()

        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)
        self.positional_encoding = LearnedPositionalEncoding(
            embed_dims//2, self.bev_h, self.bev_w)
        self.prepose = None
        self.pre_spatial_features_2d = None
        self.norm = build_norm_layer({"type": "LN"}, self.embed_dims)[1]
        self.XMIN, self.YMIN, self.ZMIN = self.point_cloud_range[:3]
        self.XMAX, self.YMAX, self.ZMAX = self.point_cloud_range[3:]
        self.prepare_coordinates_3d(point_cloud_range, voxel_size)


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


    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def translate_c3d_pre(self, feature_2d=True):
        c3d = self.coordinates_3d[0].view(-1,3).cuda().float()
        c3d[...,2] *= 0
        self.c3d = c3d

        c3d_rear = torch.concat((c3d, torch.ones((c3d.shape[0], 1)).cuda(c3d.device)),1)
        
        c3d_world = (self.ego2globals @ c3d_rear.T).T

        world2rear_pre = torch.inverse(self.ego2globals)

        c3d_rear_pre = (world2rear_pre @ c3d_world.T).T
        c3d_cam_pre = c3d_rear_pre
        return torch.tensor(c3d_cam_pre).cuda(c3d.device).float()

    def forward(self,
                inputs,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',

                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """


        (query, sensor2egos, ego2sensors, ego2globals, intrins, post_rots, post_trans, bda) = inputs
        B, C, H, W = query.shape
        assert B==1,"batch should be 1, multi batch will be supported in the future"

        self.ego2globals = ego2globals[0,0,:,:]  # x y z qw qx qy qz
        if self.pre_spatial_features_2d == None and self.prepose == None:
            self.pre_spatial_features_2d = query.detach()
            self.pre_ego2globals = self.ego2globals 
            
        c2d_cam_pre = self.translate_c3d_pre()[:, :2]
        norm_c2d_cam_pre = \
            (c2d_cam_pre - torch.as_tensor([self.XMIN, self.YMIN],  device=c2d_cam_pre.device)) /\
            torch.as_tensor(
                [self.XMAX - self.XMIN, self.YMAX - self.YMIN,
                 ],
                device=c2d_cam_pre.device)
        norm_c2d_cam_pre = norm_c2d_cam_pre * 2. - 1.
        norm_c2d_cam_pre = norm_c2d_cam_pre[:, [0, 1]].contiguous().float()

        pre_spatial_features_2d_shift = F.grid_sample(self.pre_spatial_features_2d, norm_c2d_cam_pre.view(B, H, W, -1),
                                                      align_corners=True)

        value = torch.cat([pre_spatial_features_2d_shift, query], 0)
        import cv2
        show = pre_spatial_features_2d_shift[0, 10].detach().cpu().numpy()
        cv2.imwrite('grid.jpg', (show-show.min()) /
                    (show.max()-show.min())*255)
        show = self.pre_spatial_features_2d[0, 10].detach().cpu().numpy()
        cv2.imwrite('pre_spatial_features_2d.jpg', (show-show.min()) /
                    (show.max()-show.min())*255)
        show = query[0, 10].detach().cpu().numpy()*255
        # show = (show-show.min()) / (show.max()-show.min())*255

        mask = torch.zeros((B, self.bev_h, self.bev_w),
                           device=query.device).to(query.dtype)
        query_pos = self.positional_encoding(mask)
        if query_pos is not None:
            query = query + query_pos
        query = query.permute(0, 2, 3, 1)
        query = query.view(query.shape[0], -1, query.shape[-1])
        bs,  num_query, embed_dims = query.shape
        value = value.permute(0, 2, 3, 1)
        value = value.reshape(bs*self.num_bev_queue,
                              num_query, -1)
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        value = value.reshape(bs*self.num_bev_queue,
                              num_query, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query)  # 每个采样点的偏移量
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        # 每个采样点的权重,直接通过query回归,而不是QxK,  Deformable Attention优化
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]]).to(query.device)
        reference_points = self.c3d[:, :2]
        reference_points = \
            (reference_points - torch.as_tensor([self.XMIN, self.YMIN], device=reference_points.device)) /\
            torch.as_tensor(
                [self.XMAX - self.XMIN, self.YMAX - self.YMIN],
                device=c2d_cam_pre.device)
        # reference_points = reference_points.reshape(
        #     self.num_bev_queue, num_query, 1, 2)
        reference_points = torch.cat([reference_points]*self.num_bev_queue, 0).reshape(
            self.num_bev_queue, num_query, 1, 2).float()

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        output = self.norm(output)
        output = output.permute(0, 2, 1).view(B, C, H, W)
        show = output[0, 15].detach().cpu().numpy()
        cv2.imwrite('spatial_features_2d.jpg', (show-show.min()) /
                    (show.max()-show.min())*255)
        self.pre_ego2globals = self.ego2globals 
        self.pre_spatial_features_2d = output.detach()
        return output