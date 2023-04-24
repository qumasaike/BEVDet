import numpy as np
import os

from eval_utils.bev_map_val import get_map

cnf_dict = {}
# cnf_dict['minX'] = -7.0
# cnf_dict['maxX'] =  70.6 
cnf_dict['minX'] = -20 
cnf_dict['maxX'] =  57.6 
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


def val(imagesetfile):
    input_dir = "results/dt/"
    result_dir = 'results/dt_map/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_1_f = open(result_dir+'1.txt', 'w')
    result_2_f = open(result_dir+'2.txt', 'w')
    result_3_f = open(result_dir+'3.txt', 'w')
    result_4_f = open(result_dir+'4.txt', 'w')
    result_5_f = open(result_dir+'5.txt', 'w')
    result_6_f = open(result_dir+'6.txt', 'w') 
    files = os.listdir(input_dir)
    files.sort()
    for file in files:
        # file = "0809_001253.txt"
        with open(input_dir+file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                split_txt = line[0:-1].split(' ')
                cls_id = int(split_txt[0])
                x, y, z = float(split_txt[1]), float(split_txt[2]), float(split_txt[3])
                if x < cnf_dict['minX'] or x > cnf_dict['maxX']:
                    continue
                if y < cnf_dict['minY'] or y > cnf_dict['maxY']:
                    continue
                if z < cnf_dict['minZ'] or z > cnf_dict['maxZ']:
                    continue

                l, w, h = float(split_txt[4]), float(split_txt[5]), float(split_txt[6])
                yaw = float(split_txt[7])
                score = float(split_txt[-1])

                l = l / cnf_dict['bound_size_x'] * cnf_dict['BEV_HEIGHT']
                w = w / cnf_dict['bound_size_y'] * cnf_dict['BEV_WIDTH']
                center_input_y = (x - cnf_dict['minX']) / cnf_dict['bound_size_x'] * cnf_dict['BEV_HEIGHT']  # x --> y (invert to 2D image space)
                center_input_x = (y - cnf_dict['minY']) / cnf_dict['bound_size_y'] * cnf_dict['BEV_WIDTH']  # y --> x
                yaw = -yaw
                
                bev_corners = get_corners(center_input_x, center_input_y, w, l, yaw)
                x1, y1, x2, y2, x3, y3, x4, y4 = bev_corners[0, 0], bev_corners[0, 1], bev_corners[1, 0], bev_corners[1, 1], bev_corners[2, 0], bev_corners[2, 1], bev_corners[3, 0], bev_corners[3, 1]
                
                if cls_id == 1:
                    result_1_f.write(file.split('.txt')[0]+' '+str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+'\n')
                if cls_id == 2:
                    result_2_f.write(file.split('.txt')[0]+' '+str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+'\n')
                if cls_id == 3:
                    result_3_f.write(file.split('.txt')[0]+' '+str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+'\n')
                if cls_id == 4:
                    result_4_f.write(file.split('.txt')[0]+' '+str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+'\n')
                if cls_id == 5:
                    result_5_f.write(file.split('.txt')[0]+' '+str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+'\n')
                if cls_id == 6:
                    result_6_f.write(file.split('.txt')[0]+' '+str(score)+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+str(x3)+' '+str(y3)+' '+str(x4)+' '+str(y4)+'\n')                                
        f.close()
    result_1_f.close()
    result_2_f.close()
    result_3_f.close()
    result_4_f.close()
    result_5_f.close()
    result_6_f.close()
    detpath  = result_dir+'{:s}.txt'
    # annopath = './dataset/zhijiangyihao/training2/pointcloud/test_map/{:s}.txt'
    # imagesetfile = './dataset/zhijiangyihao/training2/pointcloud/ImageSets/val_id.txt' 
    annopath = 'results/label/{:s}.txt'
    # imagesetfile = './data/zjdata/ImageSets/val0916.txt' 
    # imagesetfile = './data/zjdata/ImageSets/val0809.txt' 
    map = get_map(detpath, annopath, imagesetfile)
    print(map)

if __name__ =="__main__":
    val('datasets/zjdata/ImageSets/val.txt')