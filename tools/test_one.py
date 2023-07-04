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
import cv2
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB



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

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

def ego2img(points_ego, ego2img, intrinsic):

    points_camera_homogeneous = points_ego @ ego2img.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    points_img = points_camera @ intrinsic.T
    points_img = points_img[:, :2]
    return points_img, valid

def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid

draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                (2, 6), (3, 7)]

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
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()

    input_imgs = []
    images = []
    # for i in range(8):
    #     img = cv2.imread(os.path.join('datasets/zjdata_E1/training','image' + str(i), '0308_001055.png'))
    #     images.append(img)
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #     input_imgs.append(mmlabNormalize(img))
    # input_imgs = torch.tensor(np.stack(input_imgs)).unsqueeze(0).cuda()


    image_names = ['front_left','front_right', 'right_front', 'right_front2', 'right_rear', 'left_rear','left_front2','left_front']
    for image_name in image_names:
        img = cv2.imread(os.path.join('images','1_' + image_name + '.jpg'))
        images.append(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        input_imgs.append(mmlabNormalize(img))
    input_imgs = torch.tensor(np.stack(input_imgs)).unsqueeze(0).cuda()


    calib_path = "tools/calib_0519.pkl"
    # calib_path = "datasets/zjdata_E1/training/calib/0308_001055.pkl"
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
    model.img_view_transformer.init_coord_imgs(ego2sensors, intrinsics)
    img_inputs0 = [input_imgs,0,0,0,0,0,0,0]
    input_dict0 = (img_inputs0, None)
    boxes_3d, scores_3d, labels_3d = model(input_dict0)
    print("boxes_3d: ", boxes_3d)
    boxes_3d = boxes_3d[scores_3d>0.2]
    boxes_3d = boxes_3d.cpu()
    boxes = LB(boxes_3d, origin=(0.5, 0.5, 0.0))
    corners_ego = boxes.corners.numpy().reshape(-1, 3)
    corners_ego = np.concatenate(
        [corners_ego,
          np.ones([corners_ego.shape[0], 1])],
        axis=1)
    # draw instances
    camera_id = 2
    R_ = getattr(calib, 'R' + str(camera_id)).reshape(3,3) #rear2camera
    T_ = getattr(calib, 'T' + str(camera_id)).reshape(3,1)
    ego2sensor = np.vstack((np.hstack((R_, T_)),np.array([0.,0.,0.,1.])))
    intrinsic = getattr(calib, 'P' + str(camera_id))[:3,:3]
    corners_img, valid = ego2img(corners_ego, ego2sensor, intrinsic)
    valid = np.logical_and(
        valid,
        check_point_in_img(corners_img, img.shape[0], img.shape[1]))
    valid = valid.reshape(-1, 8)
    img = images[camera_id]
    corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
    for aid in range(valid.shape[0]):
        for index in draw_boxes_indexes_img_view:
            if valid[aid, index[0]] and valid[aid, index[1]]:
                cv2.line(
                    img,
                    corners_img[aid, index[0]],
                    corners_img[aid, index[1]],
                    color=(255, 255, 0),
                    thickness=4)
    cv2.imwrite("./vis/" + 'output.jpg',img)


if __name__ == '__main__':

    main()
