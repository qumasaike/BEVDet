import pickle
import numpy as np
p = pickle.load(open("datasets/zjdata_E1/training/calib/0308_000005.pkl",'rb'))
s = ""
for key in p['cameras']:
    print(key)
    c2r = p['cameras'][key]['camera_to_rear']
    r2c = np.linalg.inv(c2r)
    s += ",".join([str(_) for _ in r2c.reshape(-1)])+","
print(s)
print(len(s.split(',')))

s = ""
for key in p['cameras']:
    print(key)

    intrinsic = p['cameras'][key]['intrinsic']
    intr = [intrinsic['fx'], 0.0, intrinsic['cx'],0.0,intrinsic['fy'], intrinsic['cy'],0.0,0.0,1.0]
    s += ",".join([str(_) for _ in intr])+","
print(s)
print(len(s.split(',')))
