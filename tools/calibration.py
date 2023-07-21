# import torch
import numpy as np


def RotateMatirx2unitQ(m):
    w = np.sqrt(m[0,0] + m[1,1] + m[2,2] + 1)/2
    x = (m[2,1] - m[1,2]) / (4*w)
    y = (m[0,2] - m[2,0]) / (4*w)
    z = (m[1,0] - m[0,1]) / (4*w)
    return np.array([w,x,y,z])


def unproject_depth_map_to_3d(depth_map, calib, image=None):
    cu, cv = calib.cu, calib.cv
    fu, fv = calib.fu, calib.fv
    u, v = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    mask = depth_map > 0.1
    z = depth_map[mask]
    v = v[mask]
    u = u[mask]
    x = (u - cu) * z / fu
    y = (v - cv) * z / fv
    xyz = np.stack([x, y, z], -1)
    if image is not None:
        color = image[mask]
        return xyz, color, mask
    else:
        return xyz, mask


def project_points_with_mask_back_to_image(values, mask):
    u, v = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    v = v[mask]
    u = u[mask]
    output = np.zeros([mask.shape[0], mask.shape[1], *values.shape[1:]], dtype=np.float32)
    output[v, u] = values
    return output


# import torch
import numpy as np
import pickle

cameras_id = {
  'front_left':0,
  'front_right':1,
  'right_front':2,
  'right_front2':3,
  'right_rear':4,
  'left_rear':5,
  'left_front2':6,
  'left_front':7,
}



class Calibration(object):
    def __init__(self, calib_file):
        self.next = None
        self.pre = None
        if not isinstance(calib_file, dict):
            calib = self.get_calib_from_file(calib_file)
        else:
            calib = calib_file
        for camera_id in calib['CAMERAS']:
            camera_id = str(camera_id)
            for key in ['P' + camera_id, 'R' + camera_id, 'T' + camera_id]:
                assert key in calib.keys(), key + 'not in calib'
        self.__dict__.update(calib)
        self.flipped = False
        self.offsets = [0, 0]

    def get_calib_from_file(self, calib_file):
        if 'txt' in calib_file:
            with open(calib_file) as f:
                lines = f.readlines()

            calib_data = {}
            # for key in ['P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'R2_3', 'T2_3', 'dist2', 'dist3']:
            for line in lines:
                if line == '':
                    continue
                line = line.strip()
                splits = [x for x in line.split(' ') if len(x.strip()) > 0]
                obj = splits[1:]
                key = splits[0][:-1]
                if key[0] == 'P':
                    calib_data[key] = np.array(obj, dtype=np.float32).reshape(3, 4)
                elif key[0] == 'R':
                    calib_data[key] = np.array(obj, dtype=np.float32).reshape(3, 3)
                elif key[0] == 'T':
                    calib_data[key] = np.array(obj, dtype=np.float32).reshape(3)
                elif key == 'V2C':
                    calib_data[key] = np.array(obj, dtype=np.float32).reshape(3,4)
                elif key == 'CAMERAS':
                    calib_data[key] = np.array(obj, dtype=np.uint8)
                elif key == 'C2REAR':
                    calib_data[key] = np.array(obj, dtype=np.float32).reshape(3,4)
                else:
                    print('error: calib key: ', key)
        elif 'pkl' in calib_file:
            pkl = pickle.load(open(calib_file,'rb'))
            calib_data = {}
            calib_data['CAMERAS'] = []
            for camera in pkl['cameras'].keys():
                intrinsic = pkl['cameras'][camera]['intrinsic']
                camera_id = cameras_id[camera]
                calib_data['P' + str(camera_id)] = np.array([[intrinsic['fx'],0.,intrinsic['cx']],
                                                              [0, intrinsic['fy'],intrinsic['cy']],
                                                              [0.,0.,1.]])
                rear_to_camera = np.linalg.inv(pkl['cameras'][camera]['camera_to_rear'])
                calib_data['R' + str(camera_id)] = rear_to_camera[:3,:3]
                calib_data['T' + str(camera_id)] = rear_to_camera[:3,3]
                calib_data['CAMERAS'].append(camera_id)
            calib_data['lidar_to_rear'] =  pkl['lidar']['lidar_to_rear']
            self.next = pkl['next']
            self.pre = pkl['pre']
        return calib_data


    def corners3d_to_img_boxes(self, corners3d, image_id):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """

        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        P = getattr(self,'P' + str(image_id))
        R = getattr(self,'R' + str(image_id))
        T = getattr(self,'T' + str(image_id))

        corners3d_hom =  np.matmul(corners3d_hom, np.concatenate((R,T.reshape(3,1)),1).T)
        corners3d_hom = np.concatenate(
            (corners3d_hom, np.ones((sample_num, 8, 1))), axis=2)[:,:,:3]  # (N, 8, 4)
            
        img_pts = np.matmul(corners3d_hom, P[:3,:3].T)  # (N, 8, 3)
        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x[img_pts[:, :, 2]<0] = -abs(x[img_pts[:, :, 2]<0]) 
        y[img_pts[:, :, 2]<0] = -abs(y[img_pts[:, :, 2]<0]) 
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate(
            (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner
