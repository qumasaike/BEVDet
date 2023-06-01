import argparse

import torch.onnx
# from mmcv import Config
from mmengine.config import Config


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
    cfg.model.type = cfg.model.type + 'TRT'

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
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()

    for i, data in enumerate(data_loader):
        img_inputs = [[t.cuda() for t in data['img_inputs'][0]]]
        # img_metas = data['img_metas']
        img_metas = None
        input_dict = (img_inputs, img_metas)
        with torch.no_grad():
            torch.onnx.export(
                model,
                input_dict,
                args.work_dir + model_prefix + '.onnx',
                opset_version=16,
                input_names=[
                    'img_inputs', 'img_metas'
                ],
                output_names=['boxes_3d', 'scores_3d', 'labels_3d']
                              )
        break


if __name__ == '__main__':

    main()
