import os
import numpy as np
import pdb
# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


cnf_dict = {}
cnf_dict['minX'] = -20
cnf_dict['maxX'] = 57.6
cnf_dict['minY'] = -30.4
cnf_dict['maxY'] = 30.4
cnf_dict['minZ'] = -3
cnf_dict['maxZ'] = 1   
cnf_dict['BEV_HEIGHT'] = 776
cnf_dict['BEV_WIDTH'] = 608
cnf_dict['DISCRETIZATION'] = (cnf_dict['maxX'] - cnf_dict['minX']) / (cnf_dict['BEV_HEIGHT'] * 1.0)
cnf_dict['peak_thresh'] = 0.2
cnf_dict['bound_size_x'] = cnf_dict['maxX'] - cnf_dict['minX']
cnf_dict['bound_size_y'] = cnf_dict['maxY'] - cnf_dict['minY']
cnf_dict['bound_size_z'] = cnf_dict['maxZ'] - cnf_dict['minZ']

# For test camera
CLASSES = {'Car': 1, 'Van':2, 'Pedestrian': 3, 'Cyclist': 4, 'Trafficcone': 5, 'Others': 6}

val_file = "datasets/zjdata/ImageSets/val.txt"
input_dir = "datasets/zjdata/training/label"
write_dir = "./results/label/"
files = open(val_file,'r').readlines()
for file in files:
    file = file.strip() + '.txt'
    # if '0809_' not in file and '0916_' not in file:
    #     continue
    with open(os.path.join(input_dir, file), 'r') as f:
        f_w = open(write_dir+file, 'w')
        lines = f.readlines()
        for line in lines:
            split_txt = line[0:-1].split(' ')
            cls_id = CLASSES[split_txt[0]]
            # x, y, z = float(split_txt[1]), float(split_txt[2]), float(split_txt[3])
            x, y, z = float(split_txt[9]), -float(split_txt[7]), -float(split_txt[8])
            # pdb.set_trace()
            if x < cnf_dict['minX'] or x > cnf_dict['maxX']:
                continue
            if y < cnf_dict['minY'] or y > cnf_dict['maxY']:
                continue
            if z < cnf_dict['minZ'] or z > cnf_dict['maxZ']:
                continue
            # if x < -20 or x > 20:
            #     continue
            # if y < -10 or y > 10:
            #     continue
            # if z < cnf_dict['minZ'] or z > cnf_dict['maxZ']:
            #     continue
            # l,w,h = float(split_txt[4]), float(split_txt[5]), float(split_txt[6])
            h,w,l = float(split_txt[4]), float(split_txt[5]), float(split_txt[6])
            # yaw = float(split_txt[7])
            yaw = -float(split_txt[10]) - np.pi / 2
            

            # 针对是否投影到图像做box过滤
            # flag_in_image = whether_in_image([x, y, z, l, w, h, yaw])
            # if flag_in_image == False:
            #     continue


            l = l / cnf_dict['bound_size_x'] * cnf_dict['BEV_HEIGHT']
            w = w / cnf_dict['bound_size_y'] * cnf_dict['BEV_WIDTH']
            center_input_y = (x - cnf_dict['minX']) / cnf_dict['bound_size_x'] * cnf_dict['BEV_HEIGHT']  # x --> y (invert to 2D image space)
            center_input_x = (y - cnf_dict['minY']) / cnf_dict['bound_size_y'] * cnf_dict['BEV_WIDTH']  # y --> x
            yaw = -yaw
            
            bev_corners = get_corners(center_input_x, center_input_y, w, l, yaw)
            x1, y1, x2, y2, x3, y3, x4, y4 = \
            bev_corners[0, 0], bev_corners[0, 1], bev_corners[1, 0], bev_corners[1, 1], \
            bev_corners[2, 0], bev_corners[2, 1], bev_corners[3, 0], bev_corners[3, 1]
            f_w.write(str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+' '+str(cls_id)+' 0'+'\n')
    f_w.close()
    f.close()
