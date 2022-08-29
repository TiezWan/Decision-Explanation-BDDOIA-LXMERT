# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
# import glob

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from .param import args
# from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
# TINY_IMG_NUM = 512
# FAST_IMG_NUM = 5000

# The path to data and image features.
#VQA_DATA_ROOT = 'data/vqa/'
# annotations_file = '/root/Documents/ISVQA/imdb_nuscenes_test_score_id.json'
# SPLIT2NAME = {
#     'train': 'train2014',
#     'valid': 'val2014',
#     'minival': 'val2014',
#     'nominival': 'val2014',
#     'test': 'test2015',
# }


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    # def __init__(self, splits: str):
    #     self.name = splits
    #     self.splits = splits.split(',')
    def __init__(self, annotations_file):
        # Loading datasets
        self.data = []
        # for split in self.splits:
        #     self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        # print("Load %d data from split(s) %s." % (len(self.data), self.name))
        # self.data = json.load(open(annotations_file))['data']
        self.data = json.load(open(annotations_file))

        # Convert list to dict (for evaluation)
        self.id2datum = {
             datum['id']: datum
             for datum in self.data
        }

        # print(self.id2datum)

        # # Answers
        # self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        # self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        # assert len(self.ans2label) == len(self.label2ans)
        self.ans2label = {}
        self.label2ans = {}
        with open('./input/ISVQA/Annotation/answers_nuscenes_more_than_1.txt') as file:
            lines = file.read().splitlines()
            for idx, line in enumerate(lines):
                self.ans2label[line] = idx
                self.label2ans[idx] = line
        assert len(self.ans2label) == len(self.label2ans)


    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class FeatureLoader:
    def __init__(self, img_paths):
        print('Loading features files')
        self.img_data=[]
        list_paths = os.listdir(img_paths)
        for list_path in tqdm(list_paths):
            # for img_path in tqdm(os.path.join(img_paths, list_path)):
            img_path = os.path.join(img_paths, list_path)
            if img_path.endswith('.npy'):
                self.img_data.append(np.load(img_path, allow_pickle=True).item())  
   

class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset, img_data):
        super().__init__()
        self.raw_dataset = dataset

        # if args.tiny:
        #     topk = TINY_IMG_NUM
        # elif args.fast:
        #     topk = FAST_IMG_NUM
        # else:
        #     topk = None

        # # Loading detection features to img_data
        # img_data = []
        # for split in dataset.splits:
        #     # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
        #     # It is saved as the top 5K features in val2014_***.tsv
        #     load_topk = 5000 if (split == 'minival' and topk is None) else topk
        #     img_data.extend(load_obj_tsv(
        #         os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
        #         topk=load_topk))
        # img_data = []
        # # img_data_info = []
        # # img_paths = glob.glob('feature_output_train_test/*.npy')
        # img_paths = glob.glob('/root/dataset/NuScenes/feature_output_train_test/*.npy')
        # for img_path in img_paths:
        #     # if img_path[-8:-4] == 'info':
        #     #     img_data_info.append(np.load(img_path,  allow_pickle=True).item())
        #     # else:
        #         img_data.append(np.load(img_path, allow_pickle=True).item())
        #         # if len(img_data)>10000:
        #         #     break
        #             #print(img_data[0]['feature'].size())

        # Convert img list to dict
        self.imgid2img = {}
        #self.imgid2img_info = {}
        for i, img_datum in enumerate(img_data):
            if img_datum['image_id'] in self.imgid2img.keys():
            #if img_datum['image_id'] not in self.imgid2img.keys():    
                self.imgid2img[img_datum['image_id']].append(img_datum)
            else:
                self.imgid2img[img_datum['image_id']] = [img_datum]  # list
            # print('\r{}/{}'.format(i+1, len(img_data)), end='')
        # for img_datum in img_data_info:
        #     if img_datum['image_id'] in self.imgid2img_info.keys():
        #         self.imgid2img_info[img_datum['image_id']].append(img_datum)
        #     else:
        #         self.imgid2img_info[img_datum['image_id']] = [img_datum]
        # Only kept the data with loaded image features
        # i12s = []
        # for i, j in self.imgid2img.items():
        #     if len(j)==12:
        #         # print(i, len(j))
        #         i12s.append(i)
        # for m, i12 in enumerate(i12s):
        #     del self.imgid2img[i12]
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['image_id'] in self.imgid2img.keys():
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):

        datum = self.data[item]
        
        img_id = datum['image_id']
        ques_id = datum['id']     
        ques = datum['question_str']

        # Get image info
        obj_num = 0
        # img_data_info = self.imgid2img_info[img_id]  # img_info is a list
        # img_data = self.imgid2img[img_id]   # list
        # i12s = []
        # for i, j in self.imgid2img.items():
        #     if len(j)==12:
        #         print(i, len(j))
        #         i12s.append(i)
        # for m, i12 in enumerate(i12s):
        #     del self.imgid2img[i12]
        img_data = self.imgid2img[img_id]   # list
        for i in range(len(img_data)):
            obj_num += img_data[i]['num_boxes']
            if i == 0 :
                feats = torch.clone(img_data[i]['feature'])
                #b = feats
                #print(feats)
                boxes = img_data[i]['bbox'].copy()
            else:
                #print(torch.clone(img_data[i]['feature']))
                #a = torch.clone(img_data[i]['feature'])
                # print(a.equal(b))
                feats = torch.cat((feats, torch.clone(img_data[i]['feature'])), dim=0)
                

                boxes = np.concatenate((boxes, img_data[i]['bbox'].copy()))
        assert obj_num == len(boxes) == len(feats)


        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_data[0]['image_height'], img_data[0]['image_width']
        boxes = boxes.copy()
        boxes[np.where(boxes[:, 0] > 1600), 0] = 1600
        boxes[np.where(boxes[:, 2] > 1600), 2] = 1600
        boxes[np.where(boxes[:, 1] > 900), 1] = 900
        boxes[np.where(boxes[:, 3] > 900), 3] = 900
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            # print(label)
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items(): # .items() ans:key score:value
                if ans in self.raw_dataset.ans2label.keys():
                #     target[0] = score
                # else:
                    target[self.raw_dataset.ans2label[ans]] = score
                
            return ques_id, feats, torch.from_numpy(boxes), ques, target
        else:
            return ques_id, feats, torch.from_numpy(boxes), ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict): # no quesid in our project, replaced by id 
        score = 0.
        for id, ans in quesid2ans.items():
            datum = self.dataset.id2datum[id]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for id, ans in quesid2ans.items():  #id -> key, ans -> value
                result.append({
                    'id': id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

# dset = VQADataset(annotations_file)
# tset = VQATorchDataset(dset)
# tset[3]