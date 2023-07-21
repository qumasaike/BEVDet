#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import copy
import math
from mmdet.models import build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import FUSION_LAYERS
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet3d.models.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase
from mmdet3d.core.utils.norm_bbox import normalize_bbox


@FUSION_LAYERS.register_module()
class UniADTrack(MVXTwoStageDetector):
    """UniAD tracking part
    """
    def __init__(
        self, 
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        loss_cfg=None,
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
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.4,
        filter_score_thresh=0.35,
        miss_tolerance=5,
        gt_iou_threshold=0.3,
    ):
        super(UniADTrack, self).__init__(
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range



        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)   # the final one is ego query, which constantly models ego-vehicle
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w

    

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
    
    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    def forward(self,
                bev_embed,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_past_traj,
                gt_past_traj_mask,
                gt_inds,
                gt_sdc_bbox,
                gt_sdc_label,
                l2g_t,
                l2g_r_mat,
                img_metas,
                timestamp):
        """Forward funciton
        Args:
        Returns:
        """
        bev_pos, bev_embed = torch.split(bev_embed, self.embed_dims, dim=1)
        track_instances = self._generate_empty_tracks()
        # init gt instances!
        gt_instances_list = []


        gt_instances = Instances((1, 1))
        boxes = gt_bboxes_3d[0][i].tensor.to(bev_embed.device)
        # normalize gt bboxes here!
        boxes = normalize_bbox(boxes, self.pc_range)
        sd_boxes = gt_sdc_bbox[0][i].tensor.to(bev_embed.device)
        sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
        gt_instances.boxes = boxes
        gt_instances.labels = gt_labels_3d[0][i]
        gt_instances.obj_ids = gt_inds[0][i]
        gt_instances.past_traj = gt_past_traj[0][i].float()
        gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
        gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)  # boxes.shape[0] sometimes 0
        gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
        gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()

        all_query_embeddings = []
        all_matched_indices = []
        all_instances_pred_logits = []
        all_instances_pred_boxes = []
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
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
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        frame_res = {}
        frame_res["init_track_instances"] = self._generate_empty_tracks()
        frame_res["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        frame_res["track_instances"] = out_track_instances
        # all_query_embeddings: len=dec nums, N*256
        # all_matched_idxes: len=dec nums, N*2
        track_instances = frame_res["track_instances"]
        
        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        frame_res.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        return losses, out

   
    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        # bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask)[0]
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # filter out sleep querys
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]

