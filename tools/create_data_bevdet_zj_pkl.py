# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter
import os
from calibration import Calibration, RotateMatirx2unitQ
import cv2
map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
# classes = [
#  'Car', 'Truck', 'Pedestrian', 'Cyclist', 'Trafficcone', 'Others'
# ]
classes = {
 'Car':0, 'Truck':1, 'Van':1, 'Pedestrian':2, 'Cyclist':3, 'Trafficcone':4, 'Others':5
}


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):

    cam2psudeo_lidar = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
    for set_ in ['train', 'val']:
        dataset = {}
        infos = []
        ImageSets = open(os.path.join(root_path, 'ImageSets', set_ + '.txt')).readlines()
        for fil in ImageSets:
            fil = fil.strip()
            info = {}
            info['lidar_path'] = os.path.join(root_path, 'training', 'pc', fil + '.bin')
            calib = Calibration(os.path.join(root_path, 'training', 'calib_pkl', fil + '.pkl'))
            lidar2ego_translation = np.matmul(cam2psudeo_lidar, calib.lidar_to_rear)
            info['lidar2ego_translation'] = lidar2ego_translation[:,3]
            info['lidar2ego_rotation'] = RotateMatirx2unitQ(lidar2ego_translation)
            info['token'] = fil
            cams = {}
            for camera_id in calib.CAMERAS:
                cam = {}
                cam['data_path'] = os.path.join(root_path, 'training', 'image' + str(camera_id), fil + '.png')
                R_ = getattr(calib, 'R' + str(camera_id)).reshape(3,3)
                T_ = getattr(calib, 'T' + str(camera_id)).reshape(3,1)
                cam2sensor = np.vstack((np.hstack((R_, T_)),np.array([0.,0.,0.,1.])))
                sensor2psudeo_lidar = np.matmul(cam2psudeo_lidar,np.linalg.inv(cam2sensor))
                cam['sensor2ego_translation'] = sensor2psudeo_lidar[:3,3]
                cam['sensor2ego_rotation'] = RotateMatirx2unitQ(sensor2psudeo_lidar[:3,:3])
                sensor2lidar = np.linalg.inv(np.matmul(cam2sensor,calib.lidar_to_rear))
                cam['sensor2lidar_translation'] = sensor2lidar[:3,3]
                cam['sensor2lidar_rotation'] = RotateMatirx2unitQ(sensor2lidar[:3,:3])
                cam['cam_intrinsic'] = getattr(calib, 'P' + str(camera_id))[:3,:3]
                cam['ego2global_translation'] = np.array([0.,0.,0.])
                cam['ego2global_rotation'] = np.array([1.,0.,0.,0.])
                cams['image' + str(camera_id)] = cam
            info['cams'] = cams
            labels = open(os.path.join(root_path, 'training', 'label', fil + '.txt'), 'r').readlines()
            gt_boxes = []
            gt_names = []
            for label in labels:
                label = label.split(' ')
                gt_box = [float(_) for _ in label[4:10]] + [float(label[10]), 0., 0.]
                gt_boxes.append(gt_box)
                gt_names.append(label[0])
            gt_boxes = np.array(gt_boxes)[:,[5,3,4,2,1,0,6,7,8]]  #x, y, z, x_size, y_size, z_size, 0 ,yaw, 0 in pesudo lidar
            gt_boxes[:,6] += np.pi/2
            gt_boxes[:,1] *= -1
            gt_boxes[:,2] *= -1
            gt_boxes[:,2] += gt_boxes[:,5]/2
            if show:
                image_ids = ['image0','image1','image2','image3','image4','image5','image6','image7']
                for image_id in image_ids:
                    image = cv2.imread(cams[image_id]['data_path'])
                    for gt_box in gt_boxes:
                        xyz_ego = gt_box[:3]
                        ego2sensor = Quaternion(info['cams'][image_id]['sensor2ego_rotation']).inverse
                        sensor2ego_translation = info['cams'][image_id]['sensor2ego_translation']
                        xyz_cam = ego2sensor.rotate(xyz_ego - sensor2ego_translation)

                        uv = np.matmul(cams[image_id]['cam_intrinsic'],xyz_cam)
                        if uv[2]>0:
                            uv[0] /= uv[2]
                            uv[1] /= uv[2]
                            cv2.circle(image,uv[:2].astype(np.int),5,(0,0,255))
                    cv2.imwrite('show/' + image_id + cams[image_id]['data_path'].split('/')[-1],image)
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names
            info['ann_infos'] = (gt_boxes,[classes[_] for _ in gt_names])
            infos.append(info)
        dataset['infos'] = infos
        dataset['metadata'] = {'version': 'v1.0-trainval'}
        with open(os.path.join(root_path, '%s_infos_%s.pkl' % (extra_tag, set_)),'wb') as fid:
            pickle.dump(dataset, fid)

if __name__ == '__main__':
    dataset = 'zjdata_E1'
    version = 'v1.0'
    train_version = f'{version}-trainval'
    root_path = './datasets/' + dataset
    extra_tag = 'zjdet_E1'
    show = True
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0)

    # print('add_ann_infos')
    # add_ann_adj_info(root_path, extra_tag)
