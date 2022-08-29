# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import os
import numpy as np
from tqdm import tqdm
import torch

csv.field_size_limit(2147483647)
#Modified: orginal line was csv.field_size_limit(sys.maxsize) but python (and thus sys.maxsize stores numbers as llong (max value=9e18),
# C stores as long (2e9 max value), somewhere the csv module modifies C variables and produces an overflow.
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def load_obj_npy(flist, startk, topk=None):
    start_time = time.time()
    data=[]
    rejected=[]
    if topk==None:
        topk=len(flist)
    if topk==1:
        try:
            datum=np.load(flist[0], allow_pickle=True).item()
            a=[]
            for key in datum.keys():
                temp = [key, datum[key]]
                a.append(temp)
            data=a
        except Exception:
            print('unpickling error, file ignored : ', flist[0])
            rejected.append(file[0])
            #with open(args.output+"/rejected_npy.log")
    else:
        for file in tqdm(flist[startk:startk+topk]):
            try:
                datum=np.load(file, allow_pickle=True).item()
                a=[]
                for key in datum.keys():
                    temp = [key, datum[key]]
                    a.append(temp)
                data.append(a)
            except Exception:
                print('unpickling error, file ignored : ', file)
                rejected.append(file)

    elapsed_time = time.time() - start_time
    #load_perc=(len(data)/len(flist))*100
    #print(load_perc, "% of the data loaded in ", elapsed_time, "sec")
    return data, rejected
