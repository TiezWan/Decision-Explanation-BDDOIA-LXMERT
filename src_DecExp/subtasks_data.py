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
#REJECTED_FILES=['63a69964-61f539d6', '196fd10e-21e3fbdf', '572f3103-952eb72c', '10c0c72d-9892b105', '82bde0d8-0763a1c8', 
#'9c2a17dc-f6ef9a8a', 'e1004bf2-458b8a61', '4c37c9df-0a397dc5', '29f27e14-5d7a5c95', 'abf68b89-3b76fa6a', '5c25d741-b75f339c', '5c59e4c0-5145a91c', '029e1042-d985bee1']

#RANDOM_SEED=158466 #Chosen completely arbitrarily, but it should remain the same throughout a training run

LABELSJSON='BDD100k_subtasks_trainlabels.json'
QUESTION_NR=4

# SPLIT2JSON = {
#     'train': ['train_25k_images_actions', 'train_25k_images_reasons'],
#     'val': ['val_25k_images_actions', 'val_25k_images_reasons'],
#     'test': ['test_25k_images_actions', 'test_25k_images_reasons']
# }


class SubtaskDataset(Dataset):
    def __init__(self, split=str):
        #combine every needed info into a dictionary
        split=split.lstrip() #Removing leading spaces
        self.split=split #For further reference in __getitem__

        if len(split.split(","))>1:
            sys.exit('Multiple splits detected, choose one')

        print("Loading dataset", split)

        #At the time of writing this, only a subset of the total dataset is downloaded, we keep only this
        avail_data=[videoname.split(".")[0] for videoname in os.listdir(f"{BDD_DATA_ROOT}/{self.split}clips_downsampled_6fpv/")]

        #if args.dotrain==True: #No need to shuffle for evaluation
        #    random.Random(RANDOM_SEED).shuffle(avail_data)

        subtaskslabels_raw=json.load(open(f"../input/bdd100k/BDD100k_subtasks_{self.split}labels.json"))
        
        self.idx2label={}
        i=0
        for clip_id in subtaskslabels_raw.keys():
            for (framekey, framelabel) in subtaskslabels_raw[clip_id].items():
                #If the label has at least one non-None element
                if any([framelabel[i]!=None for i in range(len(framelabel))]):
                    #Converting to numpy array and then to tensor to retrieve nan (not possible from list directly)
                    framelabel=torch.tensor(np.array(framelabel, dtype=float), dtype=torch.float)
                    self.idx2label[i]=(framekey.split(".")[0], framelabel)
                    i+=1
        
        if (args.img_num==None or args.img_num>len(avail_data)):
            args.img_num=len(self.idx2label)
        else:
            self.idx2label={idx:self.idx2label[idx] for idx in range(args.img_num)}
        
        #{0: ("00091078-59817bb0_0_28_frame_1", [0, 1, 0, 1, 1, 0, 1, 0]),
        # 1: ...,
        # 2: ..., 
        # ...}
        
        #The native ids of each image in the dataset have been replaced here with a simple number ranging from 1 to img_num for access by the dataloader

        #self.feature_extractor = feature_extraction_utils.FeatureExtractor()

    def __len__(self):
        return len(self.idx2label)


    def __getitem__(self, idx=int):
        
        img_id="_".join(self.idx2label[idx][0].split("_")[:3])
        frame_nr=int(self.idx2label[idx][0].split("_")[-1])
        
        #itemframes=video2frames([BDD_DATA_ROOT+img_id+".mov"]) #list of frames
        
        #print('starting feature extraction')
        #timepre=time.time()
        
        #img_data=self.feature_extractor.extract_features(itemframes, img_id)
        
        #print('finished feature extraction')
        #print(f'extraction time: {time.time()-timepre}')
        try:
            img_data=np.load(f"{BDD_IMGFEAT_ROOT}/{self.split}clips_downsampled_6fpv/"+img_id+".npy", allow_pickle=True)[frame_nr]
        except:
            print(f"video to be removed: {img_id}")
            pdb.set_trace()
            print(img_id[-5:])
            if (img_id.split("_")[-1]==img_id.split("_")[-2]) or (img_id.split("_")[-1]<img_id.split("_")[-2]):
                os.remove(f"{BDD_DATA_ROOT}/{self.split}clips_downsampled_6fpv/"+img_id+".mov")
                print("video_removed")
            img_data=np.load(f"{BDD_IMGFEAT_ROOT}/{self.split}clips_downsampled_6fpv/"+"18285276-becaf072_10_14.npy", allow_pickle=True)
            
        
        #img_data is a 1-D array of n_frames dicts,
        #img_data[i] is a 8x2 dict, with entries : ['img_id', $list of 1 string element
        #                                          'features', $tensor of lists of floats                   
        #                                          'bbox', $array of lists of 4 floats                    
        #                                          'num_boxes', $int              
        #                                          'objects', $array of ints                    
        #                                          'cls_prob', $array of lists of floats                   
        #                                          'image_width', $int                  
        #                                          'image_height', $int ]

        #img_data=DecExpDataset.shufflerows(img_data) #performs row-wise position shuffling of the objects detected in the image

        feats=torch.tensor(img_data['features'], dtype=torch.float32)
        boxes_pre=img_data['bbox'].copy()
        boxes_pre[:, (0, 2)] /= 1289.6024 #Normalizing boxes to [0,1], mode xyxy
        boxes_pre[:, (1, 3)] /= 736.50415

        np.testing.assert_array_less(boxes_pre, 1+1e-5, verbose=True)
        np.testing.assert_array_less(-boxes_pre, 0+1e-5, verbose=True)

        boxes=torch.tensor(boxes_pre, dtype=torch.float32)

        obj_num=img_data['num_boxes']
       
        objs=torch.from_numpy(img_data['objects'])
        
        assert obj_num == objs.size()[0] == len(boxes) == len(feats), f'{obj_num}, {len(boxes)}, {len(feats)}'
        
        label=torch.tensor(self.idx2label[idx][1], dtype=torch.float32)
        
        return idx, img_id, obj_num, feats, boxes, objs, frame_nr, label
    