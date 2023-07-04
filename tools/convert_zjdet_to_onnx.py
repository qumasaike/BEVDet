import argparse

import torch.onnx
from mmcv import Config
# from mmengine.config import Config


try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
import torch
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module
from calibration import Calibration, RotateMatirx2unitQ
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdet', help='prefix of the save file name')
    parser.add_argument(
        '--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument(
        '--int8', action='store_true', help='Whether to use tensorrt int8')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if args.int8:
        assert args.fp16
    model_prefix = args.prefix
    if args.int8:
        model_prefix = model_prefix + '_int8'
    elif args.fp16:
        model_prefix = model_prefix + '_fp16'
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model['pts_bbox_head'].type = cfg.model['pts_bbox_head'].type+'TRT'
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    t = cfg.model.type
    cfg.model.type = t + 'TRT0'
    model0 = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model0, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model0.cuda()
    model0.eval()


    for i, data in enumerate(data_loader):
        img_inputs0 = [[t.cuda() for t in data['img_inputs'][0]]]
        img_metas = None
        input_dict0 = (img_inputs0, img_metas)
        calib_path = "tools/calib_0519.pkl"
        calib = Calibration(calib_path)
        ego2sensors = []
        intrinsics = []
        for camera_id in calib.CAMERAS:
            R_ = getattr(calib, 'R' + str(camera_id)).reshape(3,3) #rear2camera
            T_ = getattr(calib, 'T' + str(camera_id)).reshape(3,1)
            ego2sensor = np.vstack((np.hstack((R_, T_)),np.array([0.,0.,0.,1.])))
            intrinsic = getattr(calib, 'P' + str(camera_id))[:3,:3]
            ego2sensors.append(ego2sensor)
            intrinsics.append(intrinsic)
        ego2sensors = torch.tensor(ego2sensors).unsqueeze(0).cuda().float()
        intrinsics = torch.tensor(intrinsics).unsqueeze(0).cuda().float()
        model0.img_view_transformer.init_coord_imgs(ego2sensors, intrinsics)
        with torch.no_grad():
            torch.onnx.export(
                model0,
                input_dict0,
                args.work_dir + model_prefix + '.onnx',
                opset_version=16,
                input_names=[
                    'img_inputs', 'img_metas'
                ],
                output_names=['boxes_3d', 'scores_3d','labels_3d']     
                                           )
        break


if __name__ == '__main__':

    main()
