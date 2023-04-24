import json
import numpy as np
from val import val

class_mapping = {
  'Car':0,
  'Truck':1,
  'Pedestrian':2,
  'Cyclist':3,
  'Trafficcone':4, 
  'Others':5
}

results = json.load(open("trainzjdet_Anchor3DHead_2daug/pts_bbox/results_nusc.json", "rb"))
for name in results['results'].keys():
    lines = []
    f = open('results/dt/' + name + '.txt','w')
    for box in results['results'][name]:
        x,y,z = box['translation']
        w, l, h = box['size']
        yaw = box['rotation'][0]
        yaw = -yaw-np.pi/2
        class_id = class_mapping[box['detection_name']] + 1
        socre = box['detection_score']
        line = [class_id, x,y,z,l, w, h ,yaw , socre]
        lines.append(" ".join([str(_) for _ in line]) + "\n")
    f.writelines(lines)
    f.close()
val('datasets/zjdata/ImageSets/val.txt')

