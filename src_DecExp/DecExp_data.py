import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import random
import pdb
import sys
import time

from param import args
from utils import load_obj_npy

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
IMG_NUM = args.img_num

# The path to data and image features.
BDDOIA_DATA_ROOT = '../'
BDDOIA_IMGFEAT_ROOT = '../input/feature_output/'
REJECTED_FILES=['63a69964-61f539d6', '196fd10e-21e3fbdf', '572f3103-952eb72c', '10c0c72d-9892b105', '82bde0d8-0763a1c8', 
'9c2a17dc-f6ef9a8a', 'e1004bf2-458b8a61', '4c37c9df-0a397dc5', '29f27e14-5d7a5c95', 'abf68b89-3b76fa6a', '5c25d741-b75f339c', '5c59e4c0-5145a91c', '029e1042-d985bee1']

RANDOM_SEED=158464 #Chosen completely arbitrarily, but it should remain the same throughout a training run

SPLIT2JSON = {
    'train': 'gt_4a_21r_train',    
    'minival': 'gt_4a_21r_val',
    'val': 'gt_4a_21r_val',
    'valid': 'gt_4a_21r_val',
    'test': 'gt_4a_21r_test'
}

class DecExpDataset(Dataset):
    def __init__(self, split=str):
        #combine every needed info into a dictionary
        split=split.lstrip() #Removing leading spaces
        self.split=split #For further reference in __getitem__

        for i in split: #Checking that there are not multiple splits sent at once
            if i==' ' or i==',':
                sys.exit('Multiple splits detected, choose one')

        print("Loading dataset", split)

        temp=[]
        temp.extend(json.load(open("../input/%s.json" % SPLIT2JSON[split])).items()) #loading annotations
        #[('00067cfb-e535423e', {'actions': [0, 1, 0, 0], 'reason': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]}), ...]

        data=[]
        for i in range(len(temp)): #The image samples are derived from videos by sampling it so we need to account for it in annotations
            if temp[i][0] in REJECTED_FILES: #Some files are causing issues with their 4th part
                #for j in range(4):
                #    data.append((temp[i][0]+"_"+str(j), temp[i][1]))
                pass
            else:
                for j in range(5):
                    data.append((temp[i][0]+"_"+str(j), temp[i][1]))

        if args.dotrain==True: #Evaluation calls the Dataset class for each model -> time-consuming. Only shuffle in the DataLoader in that case
            random.Random(RANDOM_SEED).shuffle(data)

        self.idx2label={}
        if args.img_num==None:
            args.img_num=len(data)
        for i in range(min(args.img_num, len(data))):
            self.idx2label[i]=(data[i][0], data[i][1]['actions']+data[i][1]['reason'])
            #{ 0:('00067cfb-e535423e_0', [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]),
            #  1:('00067cfb-e535423e_1', [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]),
            #  ...}
        del data, temp #clearing data and temp for memory optimization as all of their information is contained in self.idx2label


    def __len__(self):
        return len(self.idx2label)


    def __getitem__(self, idx=int):
        img_id=self.idx2label[idx][0]
        #Finish importing the remaining data : features, bboxes etc
        img_dir=[BDDOIA_IMGFEAT_ROOT+self.split+'/'+img_id+".npy"]
        img_data=load_obj_npy(img_dir, startk=0, topk=1)[0]

        #img_data is a 8x2 object, with entries : ['image_id', $list of 1 string element
        #                                          'feature', $tensor of lists of floats                   
        #                                          'bbox', $array of lists of 4 floats                    
        #                                          'num_boxes', $int              
        #                                          'objects', $array of ints                    
        #                                          'cls_prob', $list of classification probabilities                    
        #                                          'image_width', $list of ints                   
        #                                          'image_height', $list of ints ]

        img_data=DecExpDataset.shufflerows(img_data) #performs row-wise position shuffling of the objects detected in the image

        self.idx2imgdata={}
        self.idx2imgdata[idx] = img_data
        
        #feats=torch.narrow(torch.tensor(img_data[1][1], dtype=torch.float32), 1, 0, 100)
        feats=torch.tensor(img_data[1][1], dtype=torch.float32)
        boxes=img_data[2][1].copy()
        boxes[:, (0, 2)] /= 1289.6024 #Normalizing boxes to [0,1], mode xyxy
        boxes[:, (1, 3)] /= 736.50415
        np.testing.assert_array_less(boxes, 1+1e-5, verbose=True)
        np.testing.assert_array_less(-boxes, 0+1e-5, verbose=True)
        boxes=torch.tensor(boxes, dtype=torch.float32)

        obj_num=img_data[3][1]

        assert obj_num == len(boxes) == len(feats)

        label=torch.tensor(self.idx2label[idx][1], dtype=torch.float32)

        return idx, img_id, obj_num, feats, boxes, label
    
    def shufflerows(image_data):
        shufflevector=torch.randperm(100)
        image_data[1][1]=image_data[1][1][shufflevector]
        image_data[2][1]=image_data[2][1][shufflevector]
        image_data[4][1]=image_data[4][1][shufflevector]
        image_data[5][1]=image_data[5][1][shufflevector]
        return image_data


