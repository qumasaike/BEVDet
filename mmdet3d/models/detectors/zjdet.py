# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class ZJDet(MVXTwoStageDetector):
# class ZJDet(CenterPoint):
  
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, img_backbone, img_neck, img_view_transformer,voxel_feature_encoder, pts_bbox_head, train_cfg, test_cfg, **kwargs):
        super(ZJDet, self).__init__(**kwargs)
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.voxel_feature_encoder = builder.build_voxel_encoder(voxel_feature_encoder)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        # self.dense_head = builder.build_neck(dense_head)


    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        # x = [imgs] + list(x)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        img_H,img_W = img[0].shape[-2:]
        x = self.image_encoder(img[0])
        x = self.img_view_transformer([x] + img[1:8] + [(img_H,img_W)])
        x = self.voxel_feature_encoder(x)
        if not type(x) is list:
            x = [x]
        return x, None


    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        # img_H,img_W = img_inputs[0].shape[-2:]
        # img_feats = self.extract_img_feat(img_inputs, img_metas)
        # x = self.image_encoder(img_inputs[0])
        # x = self.img_view_transformer([x] + img_inputs[1:7] + [(img_H,img_W)])
        # img_feats = self.voxel_feature_encoder(x)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            # img_H,img_W = img_inputs[0][0].shape[-2:]
            # # img_feats = self.extract_img_feat(img_inputs, img_metas)
            # x = self.image_encoder(img_inputs[0][0])
            # x = self.img_view_transformer([x] + img_inputs[0][1:7] + [(img_H,img_W)])
            # img_feats = self.voxel_feature_encoder(x)
            img_feats, pts_feats, _ = self.extract_feat(
                points, img=img_inputs[0], img_metas=img_metas, **kwargs)
            bbox_list = [dict() for _ in range(len(img_metas))]
            bbox_pts = self.simple_test_pts(img_feats, img_metas[0], rescale=False)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
            return bbox_list
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs

 
@DETECTORS.register_module()
class ZJDetTRT(ZJDet):
    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img_inputs=None,
        img_metas=None,
        points=None,
        proposals=None,
        gt_bboxes_ignore=None,
        **kwargs
    ):
        from mmdet3d.core.bbox import get_box_type
        box_type_3d, box_mode_3d = get_box_type("LiDAR")
        img_metas = [[{
          'flip':False,
          'pcd_horizontal_flip':False,
          'pcd_vertical_flip':False,
          'box_mode_3d':box_mode_3d,
          'box_type_3d':box_type_3d,
          'pcd_scale_factor':1.0,
          'pts_filename':""
        }]]
        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img_inputs[0], img_metas=img_metas, **kwargs)
        # bbox_list = [dict() for _ in range(len(img_metas))]
        # bbox_pts = self.simple_test_pts(img_feats, img_metas[0], rescale=False)
        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        # return bbox_list

        # bbox_list = [dict() for _ in range(len(img_metas))]
        result_dict = {}
        bbox_pts = self.simple_test_pts(img_feats, img_metas[0], rescale=False)
        for pts_bbox in bbox_pts:
            # result_dict['pts_bbox'] = pts_bbox
            boxes_3d, scores_3d, labels_3d = \
               pts_bbox['boxes_3d'].tensor, pts_bbox['scores_3d'],pts_bbox['labels_3d']
        return boxes_3d, scores_3d, labels_3d

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)