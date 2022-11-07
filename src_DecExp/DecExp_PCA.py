import os
import collections
import sys
import shutil

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import datetime
import numpy as np
import pdb #For debugging
from sklearn.decomposition import IncrementalPCA

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from DecExp_data import DecExpDataset

os.chdir("/root/Documents/ISVQA-main/src_DecExp/")
trainimgs=os.listdir("../input/lastframe/data/train")
testimgs=os.listdir("../input/lastframe/data/test")
valimgs=os.listdir("../input/lastframe/data/val")
lastframeimgs=trainimgs+testimgs+valimgs

bdd100imgs=os.listdir("../input/bdd100k/images/100k/train")+os.listdir("../input/bdd100k/images/100k/test")+os.listdir("../input/bdd100k/images/100k/val")
#for i in bdd100imgs:
#    if i not in lastframeimgs:
#        try:
#            shutil.copy("../input/bdd100k/images/100k/train/"+i, "../input/bdd100k_cleaned/images/100k/train/"+i)
#        except:
#            try:
#                shutil.copy("../input/bdd100k/images/100k/test/"+i, "../input/bdd100k_cleaned/images/100k/test/"+i)
#            except:
#                try:
#                    shutil.copy("../input/bdd100k/images/100k/val/"+i, "../input/bdd100k_cleaned/images/100k/val/"+i)
#                except:
#                    print("error")
                    
bdd100cleanimgs=os.listdir("../input/bdd100k_cleaned/images/100k/train")+os.listdir("../input/bdd100k_cleaned/images/100k/test")+os.listdir("../input/bdd100k_cleaned/images/100k/val")

verif=[]
for i in bdd100imgs:
    if i in lastframeimgs:
        verif.append(i)
print(len(verif))