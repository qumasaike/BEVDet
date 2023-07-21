# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-26.8, -32.0, -3.0, 50.0, 32.0, 1.0]
point_cloud_range = [-32.0, -26.8, -3.0, 32.0, 50.0,  1.0]
voxel_size = [0.2, 0.2, 0.2]
bev_w_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
bev_h_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
numC_Trans = 80
# For nuScenes we usually do 10-class detection
class_names = [
    'Car', 'Van', 'Pedestrian', 'Cyclist', 'Trafficcone', 'Others'
]

data_config = {
    'cams': [
        'image0','image1','image2','image3','image4','image5','image6','image7'
    ],
    'Ncams':
    8,
    'input_size': (720, 1280),#(256, 704),
    'src_size': (720, 1280),

    'resize': (-0.06, 0.11),#1.0 + resize
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model




# self.module_topology = [
#     'backbone', 'gridsample', 'voxel_feature', 'temporal_self_attention', 'dense_head_2d', 'dense_head'
# ]

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
fut_steps = 4
past_steps = 4

model = dict(
    type='ZJDetTrack',
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=32,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='GridSample',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        ),
    voxel_feature_encoder=dict(
        type='VoxelFeature',
        # input_channels=160,
        # output_channels=64,
        input_channels=640,
        output_channels=512,
        Ncams=data_config['Ncams'],
        gn=True),
    temporal_fusion=dict(
        type='TemporalSelfAttention',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        embed_dims=512),

    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=past_steps,
        fut_steps=fut_steps,
        loss_cfg=dict(
            type="ClipMatcher",
            num_classes=10,
            weight_dict=None,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3DTrack",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                pc_range=point_cloud_range,
            ),
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_past_traj_weight=0.0,
        ),  # loss cfg for tracking
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),


    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[bev_w_, bev_h_, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'datasets/zjdata_E1/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0,)

    # rot_lim=(-22.5, 22.5),
    # scale_lim=(0.95, 1.05),
    # flip_dx_ratio=0.5,
    # flip_dy_ratio=0.5),

    # 'resize': (0.00, 0.00),#(-0.06, 0.11),#1.0 + resize
    # 'rot': (0, 0),  #(-5.4, 5.4),
    # 'flip': False, #True,
    # 'crop_h': (0.0, 0.0),
    # 'resize_test': 0.00,

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False,
        # ego_cam='image0',
        ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds_3d'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config,
        sequential=False,
        # ego_cam='image0',
        ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    data_root=data_root,
    pipeline=test_pipeline,
    classes=class_names,
    ann_file=data_root + 'zjdet_E1_infos_val.pkl')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    shuffle=False,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'zjdet_E1_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=4,
#     train=dict(
#         type='CBGSDataset',
#         dataset=dict(
#         data_root=data_root,
#         ann_file=data_root + 'zjdet_E1_infos_train.pkl',
#         pipeline=train_pipeline,
#         classes=class_names,
#         test_mode=False,
#         use_valid_flag=False,
#         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#         box_type_3d='LiDAR')),
#     val=test_data_config,
#     test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)


# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# unstable
# fp16 = dict(loss_scale='dynamic')
