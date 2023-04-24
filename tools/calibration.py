# import torch
import numpy as np


def RotateMatirx2unitQ(m):
    w = np.sqrt(m[0,0] + m[1,1] + m[2,2] + 1)/2
    x = (m[2,1] - m[1,2]) / (4*w)
    y = (m[0,2] - m[2,0]) / (4*w)
    z = (m[1,0] - m[0,1]) / (4*w)
    return np.array([w,x,y,z])

def get_calib_from_file(calib_file):
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


    return calib_data


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file
        for camera_id in calib['CAMERAS']:
            camera_id = str(camera_id)
            for key in ['P' + camera_id, 'R' + camera_id, 'T' + camera_id]:
                assert key in calib.keys(), key + 'not in calib'
        self.__dict__.update(calib)

        # self.P2 = calib['P2']  # 3 x 4
        # self.P3 = calib['P3']  # 3 x 4
        # self.R0 = calib['R0']  # 3 x 3
        # self.V2C = calib['Tr_velo2cam']  # 3 x 4
        # self.R2_3 = calib['R2_3']  # 3 x 4
        # self.T2_3 = calib['T2_3']  # 3 x 4
        # self.dist2 = calib['dist2']  # 3 x 4
        # self.dist3 = calib['dist3']  # 3 x 3
        self.flipped = False
        self.offsets = [0, 0]

    @property
    def cu(self):
        return self.P2[0, 2]

    @property
    def cv(self):
        return self.P2[1, 2]

    @property
    def fu(self):
        return self.P2[0, 0]

    @property
    def fv(self):
        return self.P2[1, 1]

    @property
    def tx(self):
        return self.P2[0, 3] / (-self.fu)

    @property
    def ty(self):
        return self.P2[1, 3] / (-self.fv)

    @property
    def txyz(self):
        return np.matmul(np.linalg.inv(self.P2[:3, :3]), self.P2[:3, 3:4]).squeeze(-1)

    @property
    def K(self):
        return self.P2[:3, :3]

    @property
    def K3x4(self):
        return np.concatenate([self.P2[:3, :3], np.zeros_like(self.P2[:3, 3:4])], axis=1)

    @property
    def inv_K(self):
        return np.linalg.inv(self.K)

    def global_scale(self, scale_factor):
        self.P2[:, 3:4] *= scale_factor
        self.P3[:, 3:4] *= scale_factor

    def scale(self, scale_factor):
        self.P2[:2, :] *= scale_factor
        self.P3[:2, :] *= scale_factor

    def offset(self, offset_x, offset_y):
        K = self.K.copy()
        inv_K = self.inv_K
        T2 = np.matmul(inv_K, self.P2)
        T3 = np.matmul(inv_K, self.P3)
        K[0, 2] -= offset_x
        K[1, 2] -= offset_y
        self.P2 = np.matmul(K, T2)
        self.P3 = np.matmul(K, T3)
        self.offsets[0] += offset_x
        self.offsets[1] += offset_y

    def fliplr(self, image_width):
        # mirror using y-z plane of cam 0
        assert not self.flipped

        K = self.P2[:3, :3].copy()
        inv_K = np.linalg.inv(K)
        T2 = np.matmul(inv_K, self.P2)
        T3 = np.matmul(inv_K, self.P3)
        T2[0, 3] *= -1
        T3[0, 3] *= -1

        K[0, 2] = image_width - 1 - K[0, 2]
        self.P3 = np.matmul(K, T2)
        self.P2 = np.matmul(K, T3)

        # delete useless parameters to avoid bugs
        del self.R0, self.V2C

        self.flipped = not self.flipped

    @property
    def fu_mul_baseline(self):
        return np.abs(self.P2[0, 3] - self.P3[0, 3])

    @staticmethod
    def cart_to_hom(pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack(
            (pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        if self.flipped:
            raise NotImplementedError

        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack(
            (self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack(
            (R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack(
            (self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(
            np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    @staticmethod
    def rect_to_lidar_pseudo(pts_rect):
        pts_rect_hom = Calibration.cart_to_hom(pts_rect)
        T = np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        if self.flipped:
            raise NotImplementedError
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        # pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        pts_rect = np.dot(pts_lidar_hom, self.V2C.T)
        pts_rect = np.dot(pts_rect, self.R0.T)
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    @staticmethod
    def lidar_pseudo_to_rect(pts_lidar):
        pts_lidar_hom = Calibration.cart_to_hom(pts_lidar)
        T = np.array([[0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 0]], dtype=np.float32)
        pts_rect = np.dot(pts_lidar_hom, T)
        return pts_rect

    # def torch_lidar_pseudo_to_rect(self, pts_lidar):
    #     pts_lidar_hom = torch.cat([pts_lidar, torch.ones_like(pts_lidar[..., -1:])], dim=-1)
    #     T = np.array([[0, 0, 1],
    #                   [-1, 0, 0],
    #                   [0, -1, 0],
    #                   [0, 0, 0]], dtype=np.float32)
    #     T = torch.from_numpy(T).cuda()
    #     pts_rect = torch.matmul(pts_lidar_hom, T)
    #     return pts_rect

    def rect_to_img(self, pts_rect, image_id):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        # pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_2d_hom = np.dot(pts_rect_hom, self.__getattribute__('P' + image_id).T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - \
            self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    # def torch_rect_to_img(self, pts_rect):
    #     """
    #     :param pts_rect: (N, 3)
    #     :return pts_img: (N, 2)
    #     """
    #     pts_rect_hom = torch.cat([pts_rect, torch.ones_like(pts_rect[..., -1:])], dim=-1)
    #     pts_2d_hom = torch.matmul(pts_rect_hom, torch.from_numpy(self.P2.T).cuda())
    #     pts_img = pts_2d_hom[..., 0:2] / pts_rect_hom[..., 2:3]
    #     return pts_img

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        if self.flipped:
            raise NotImplementedError
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

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
            (corners3d_hom, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)
            
        img_pts = np.matmul(corners3d_hom, P.T)  # (N, 8, 3)
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
