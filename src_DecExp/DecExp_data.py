from cmath import inf
import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset
import random
import pdb
import sys
import time
import feature_extraction_utils

from param import args
from utils import load_obj_npy, video2frames, list2Tensor

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
IMG_NUM = args.img_num

# The path to data and image features.
BDD_DATA_ROOT = '../input/bdd100k'
BDD_IMGFEAT_ROOT = '../input/bdd100k/feature_output'
NBFRAMESAMPLES=6
WORDSET_ROOT='../input/bdd100k/wordset.txt'
#REJECTED_FILES=['63a69964-61f539d6', '196fd10e-21e3fbdf', '572f3103-952eb72c', '10c0c72d-9892b105', '82bde0d8-0763a1c8', 
#'9c2a17dc-f6ef9a8a', 'e1004bf2-458b8a61', '4c37c9df-0a397dc5', '29f27e14-5d7a5c95', 'abf68b89-3b76fa6a', '5c25d741-b75f339c', '5c59e4c0-5145a91c', '029e1042-d985bee1']

RANDOM_SEED=158466 #Chosen completely arbitrarily, but it should remain the same throughout a training run


# SPLIT2JSON = {
#     'train': ['train_25k_images_actions', 'train_25k_images_reasons'],
#     'val': ['val_25k_images_actions', 'val_25k_images_reasons'],
#     'test': ['test_25k_images_actions', 'test_25k_images_reasons']
# }

LABELSJSON='BDD100k_id2frameslabels.json'

class DecExpDataset(Dataset):
    def __init__(self, split=str):
        #combine every needed info into a dictionary
        split=split.lstrip() #Removing leading spaces
        self.split=split #For further reference in __getitem__

        if len(split.split(","))>1:
            sys.exit('Multiple splits detected, choose one')

        print("Loading dataset", split)

        #At the time of writing this, only a subset of the total dataset is downloaded, we keep only this
        avail_data=[videoname.split(".")[0] for videoname in os.listdir(f"{BDD_DATA_ROOT}/{self.split}clips_downsampled_6fpv/")]

        if args.dotrain==True: #No need to shuffle for evaluation
            random.Random(RANDOM_SEED).shuffle(avail_data)

        id2frameslabels_raw=json.load(open(f"../input/bdd100k/{LABELSJSON}"))
        
        if (args.img_num==None or args.img_num>len(avail_data)):
            args.img_num=len(avail_data)
        
        errorflag=False
        self.idx2label={}
        
        for key in range(args.img_num):
            try:
                self.idx2label[key]=[avail_data[key].split(".")[0]]
                self.idx2label[key].extend(id2frameslabels_raw[avail_data[key].split("_")[0]][avail_data[key].split("_")[1]][1:])
            except:
                print(f"video in input folder not in labels :{avail_data[key]}")
                os.remove(f"{BDD_DATA_ROOT}/{self.split}clips_downsampled_6fpv/"+avail_data[key]+".mov") 
                errorflag=True
                pass
        if errorflag:
            sys.exit()
        
        #{0: ['167f4689-bd993700_21_22', [0, 1, 0, 0], 'The car is now stopping', 'because the stoplight changed to red'],
        # 1: ['0464d3e2-8844a22c_0_16', [0, 1, 0, 0], 'The car is stopped', 'because there is a red light.'],
        # 2: ..., 
        # ...}
        
        #The native ids of each image in the dataset have been replaced here with a simple number ranging from 1 to img_num for access by the dataloader

        self.feature_extractor = feature_extraction_utils.FeatureExtractor()


    def __len__(self):
        return len(self.idx2label)


    def __getitem__(self, idx=int):
        
        img_id=self.idx2label[idx][0]
        
        #itemframes=video2frames([BDD_DATA_ROOT+img_id+".mov"]) #list of frames
        
        #print('starting feature extraction')
        #timepre=time.time()
        
        #img_data=self.feature_extractor.extract_features(itemframes, img_id)
        
        #print('finished feature extraction')
        #print(f'extraction time: {time.time()-timepre}')
        try:
            img_data=np.load(f"{BDD_IMGFEAT_ROOT}/{self.split}clips_downsampled_6fpv/"+img_id+".npy", allow_pickle=True)
        except:
            print(f"video to be removed: {img_id}")
            print(img_id[-5:])
            if (img_id.split("_")[-1]==img_id.split("_")[-2]) or (img_id.split("_")[-1]<img_id.split("_")[-2]):
                os.remove(f"{BDD_DATA_ROOT}/{self.split}clips_downsampled_6fpv/"+img_id+".mov")
                print("video_removed")
            img_data=np.load(f"{BDD_IMGFEAT_ROOT}/{self.split}clips_downsampled_6fpv/"+"18285276-becaf072_10_14.npy", allow_pickle=True)
            
        
        #img_data is a 8x2 dict, with entries : ['image_id', $list of 1 string element
        #                                          'feature', $tensor of lists of floats                   
        #                                          'bbox', $array of lists of 4 floats                    
        #                                          'num_boxes', $int              
        #                                          'objects', $array of ints                    
        #                                          'cls_prob', $array of lists of floats                   
        #                                          'image_width', $int                  
        #                                          'image_height', $int ]

        #img_data=DecExpDataset.shufflerows(img_data) #performs row-wise position shuffling of the objects detected in the image
        nbframes=len(img_data)
        self.idx2imgdata={}
        self.idx2imgdata[idx] = img_data

        tempfeats=[torch.tensor(img_data[i]['features'], dtype=torch.float32) for i in range(nbframes)]
        boxes_pre=[img_data[i]['bbox'].copy() for i in range(nbframes)]
        for i in range(nbframes):
            boxes_pre[i][:, (0, 2)] /= 1289.6024 #Normalizing boxes to [0,1], mode xyxy
            boxes_pre[i][:, (1, 3)] /= 736.50415

            np.testing.assert_array_less(boxes_pre[i], 1+1e-5, verbose=True)
            np.testing.assert_array_less(-boxes_pre[i], 0+1e-5, verbose=True)

        tempboxes=[torch.tensor(box, dtype=torch.float32) for box in boxes_pre]

        obj_num=[img_data[i]['num_boxes'] for i in range(nbframes)]
       
        tempobjs=[torch.from_numpy(img_data[i]['objects']) for i in range(nbframes)]
        
        assert obj_num[0] == tempobjs[0].size()[0] == len(tempboxes[0]) == len(tempfeats[0]), f'{obj_num[0]}, {len(tempboxes[0])}, {len(tempfeats[0])}'
        
        labeldigits=torch.tensor(self.idx2label[idx][1], dtype=torch.float32)
        label=[self.idx2label[idx][2], self.idx2label[idx][3]]
        
        #convert lists to tensors
        feats=list2Tensor(tempfeats)
        boxes=list2Tensor(tempboxes)
        objs=list2Tensor(tempobjs)
        
        if feats.size()[0]!=NBFRAMESAMPLES:
            assert boxes.size()[0]!=NBFRAMESAMPLES
            assert objs.size()[0]!=NBFRAMESAMPLES
            feats=torch.cat([feats, feats[-1, :, :].repeat(NBFRAMESAMPLES-nbframes, 1, 1)], dim=0)
            boxes=torch.cat([boxes, boxes[-1, :, :].repeat(NBFRAMESAMPLES-nbframes, 1, 1)], dim=0)
            objs=torch.cat([objs, objs[-1, :].repeat(NBFRAMESAMPLES-nbframes, 1)], dim=0)
            obj_num.extend([obj_num[-1]]*(NBFRAMESAMPLES-nbframes))
            
        return idx, img_id, obj_num, feats, boxes, objs, labeldigits, label
    
    def shufflerows(image_data):
        shufflevector=torch.randperm(100)
        image_data[1][1]=image_data[1][1][shufflevector]
        image_data[2][1]=image_data[2][1][shufflevector]
        image_data[4][1]=image_data[4][1][shufflevector]
        image_data[5][1]=image_data[5][1][shufflevector]
        return image_data


