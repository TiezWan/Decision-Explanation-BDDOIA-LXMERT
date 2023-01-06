# from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pdb
from utils import video2frames
from PIL import Image

# f=open('snap/no_clweight_reg_0p0003_train_5000_12h_3_3_3_bs8/traintrace.log', 'r')
# trainlist=[]
# for i in range(22):
#     trainlist.append(f.readline()[5:-2].split(', '))
#     for j in range(25):
#         trainlist[i][j]=int(trainlist[i][j])

# f=open('snap/no_clweight_reg_0p0003_train_5000_12h_3_3_3_bs8/valtrace.log', 'r')
# vallist=[]
# for i in range(22):
#     vallist.append(f.readline()[5:-2].split(', '))
#     for j in range(25):
#         vallist[i][j]=int(vallist[i][j])

# epochs=[i for i in range(1,23)]
# classes=[i for i in range(1,26)]

# plt.bar(classes, trainlist[0])
# plt.show()


im_folders="../input/bdd100k/valclips_downsampled_6fpv/"
path="../input/bdd100k/feature_output/valclips_downsampled_6fpv/"
list_feats=os.listdir(path)
list_folders=os.listdir(im_folders)

# for i in list_folders: #for each val video
#     if not os.path.isdir(im_folders+i.split(".")[0]+"/"):
#         os.makedirs(im_folders+i.split(".")[0]+"/")
#         os.rename(im_folders+i, im_folders+i.split(".")[0]+"/"+i)
# pdb.set_trace()

for folder in list_folders:
    for item in os.listdir(im_folders+folder):
        if item.split(".")[-1]=="jpeg":
            os.remove(im_folders+folder+"/"+item)
pdb.set_trace()            


for folder in list_folders:
    for vid in os.listdir(im_folders+folder):
        frames=video2frames([im_folders+folder+"/"+vid])
        for i, frame in enumerate(frames[0]):
            frame=np.moveaxis(frame, -1, 0)
            frame[[2, 0]] = frame[[0, 2]] #permute colors. Source : BGR, objective : RGB
            frame=np.moveaxis(frame, 0, -1)
            im = Image.fromarray(frame.astype(np.uint8))
            im.save(im_folders+folder+"/"+vid.split(".")[0]+"_frame_"+str(i)+".jpeg")
            
pdb.set_trace()

print(list_feats[0])
feats=np.load(im_folders+list_feats[0], allow_pickle=True)
cats=json.load(open("visual_genome_categories.json"))
print(cats.keys())
objects=[]
for i in feats:
    temp=[]
    for object in i['objects']:
        temp.append(cats["categories"][object]['name'])
    objects.append(temp)
print(objects)
