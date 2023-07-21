#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# Modified from bevformer (https://github.com/fundamentalvision/BEVFormer)        #
#---------------------------------------------------------------------------------#

import copy
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
# from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.utils.norm_bbox import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.track_head_plugin import MemoryBank, QueryInteractionModule, Instances
from mmdet.models import build_loss
import numpy as np
from ..builder import HEADS

@HEADS.register_module()
class BEVFormerTrackHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 past_steps=4,
                 fut_steps=4,
                 loss_cfg=None,
                 embed_dims=256,
                 qim_args=dict(
                      qim_type="QIMBase",
                      merger_dropout=0,
                      update_query_pos=False,
                      fp_ratio=0.3,
                      random_drop=0.1,
                  ),
                 mem_args=dict(
                      memory_bank_type="MemoryBank",
                      memory_bank_score_thresh=0.0,
                      memory_bank_len=4,
                  ),
                 gt_iou_threshold=0.3,
                 voxel_size=[0.2, 0.2, 0.2],
                 point_cloud_range=[-32.0, -26.8, -3.0, 32.0, 50.0,  1.0],
                 **kwargs):

        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(self.grid_size).astype(np.int64)

        self.bev_w, self.bev_h = self.grid_size[0], self.grid_size[1]


        self.fp16_enabled = False

        self.with_box_refine = with_box_refine

        assert as_two_stage is False, 'as_two_stage is not supported yet.'
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        super(BEVFormerTrackHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.embed_dims = embed_dims
        self.criterion = build_loss(loss_cfg)
        self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)   # the final one is ego query, which constantly models ego-vehicle

        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.mem_bank_len = mem_args["memory_bank_len"]
        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        self.gt_iou_threshold = gt_iou_threshold
        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
    
    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )
        return bev_embed, bev_pos

    def get_detections(
        self, 
        bev_embed,
        object_query_embeds=None,
        ref_points=None,
        img_metas=None,
    ):
        bev_embed = bev_embed.squeeze(0).permute(1,2,0)
        bev_embed = bev_embed.view(-1, bev_embed.shape[2])
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points=ref_points,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # reference = init_reference
                reference = ref_points.sigmoid()
            else:
                reference = inter_references[lvl - 1]
                # ref_size_base = inter_box_sizes[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])  # xydxdyxdz
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            last_ref_points = torch.cat(
                [tmp[..., 0:2], tmp[..., 4:5]], dim=-1,
            )

            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # tmp[..., 2:4] = tmp[..., 2:4] + ref_size_basse[..., 0:2]
            # tmp[..., 5:6] = tmp[..., 5:6] + ref_size_basse[..., 2:3]

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        last_ref_points = inverse_sigmoid(last_ref_points)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'last_ref_points': last_ref_points,
            'query_feats': hs,
        }
        return outs
        
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            bbox_index = preds['bbox_index']
            mask = preds['mask']

            ret_list.append([bboxes, scores, labels, bbox_index, mask])

        return ret_list

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict
    



    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def forward(self,                
                bev_embed,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_inds,
                img_metas,
                ):
        """Forward funciton
        Args:
        Returns:
        """
        bev_pos, bev_embed = torch.split(bev_embed, self.embed_dims, dim=1)
        track_instances = self._generate_empty_tracks()
        # init gt instances!
        gt_instances_list = []


        gt_instances = Instances((1, 1))
        boxes = gt_bboxes_3d[0].tensor.to(bev_embed.device)
        # normalize gt bboxes here!
        boxes = normalize_bbox(boxes, self.pc_range)

        gt_instances.boxes = boxes
        gt_instances.labels = gt_labels_3d[0] 
        gt_instances.obj_ids = gt_inds[0]
        gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()

        all_query_embeddings = []
        all_matched_indices = []
        all_instances_pred_logits = []
        all_instances_pred_boxes = []
        det_output = self.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        frame_res = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]

        ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]

            frame_res["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                frame_res, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        frame_res.update(self.select_active_track_query(track_instances, active_index, img_metas))
        
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        frame_res["track_instances"] = out_track_instances
        # all_query_embeddings: len=dec nums, N*256
        # all_matched_idxes: len=dec nums, N*2
        track_instances = frame_res["track_instances"]
        
        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                   ]
        out.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        return losses, out

