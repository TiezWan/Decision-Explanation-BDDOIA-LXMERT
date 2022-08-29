import numpy as np
import os
from pathlib import Path

path = '/root/dataset/NuScenes/'
feature_path = ''
parts = os.listdir(path)
ps = []
for p in parts:
    if p[:4]== 'part':
        ps.append(p)
for pss in ps:
# if os.path.exists(path + p + '/samples/' + 'n008-2018-09-18-14-54-39-0400__CAM_FRONT_LEFT__1537297176104799.jpg' for p in parts if p[:4]== 'part'):
    if Path(path + p + '/samples/CAM_FRONT_LEFT/' + 'n008-2018-09-18-14-54-39-0400__CAM_FRONT_LEFT__1537297176104799.jpg').exists: 
        print(pss)
pass
