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
import torch.nn as nn
import pdb
import cv2


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


def load_obj_npy(flist, startk=0, topk=None):
    #start_time = time.time()
    data=[]
    rejected=[]
    if topk==None:
        topk=len(flist)
    if topk==1:
        try:
            pdb.set_trace()
            datum=np.load(flist[0], allow_pickle=True)
            ##Transform the dict into a list
            a=[]
            for i in range(len(datum)):
               for key in datum[i].keys():
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

    #elapsed_time = time.time() - start_time
    #load_perc=(len(data)/len(flist))*100
    #print(load_perc, "% of the data loaded in ", elapsed_time, "sec")
    return data, rejected


def video2frames(videolist, writepath=None):
    for video in videolist:
        vidcap = cv2.VideoCapture(video)
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"number of frames: {num_frames}")
        v_w  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # int `width`
        v_h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # int `height`
        videos=[]
        frames=np.zeros([num_frames, v_h, v_w, 3])

        for image_num in range(0, int(num_frames)):
            #image=cv2.flip(vidcap.read()[1], -1) #height and width are confused at reading, so a flip is necessary
            image=vidcap.read()[1]
            frames[image_num,:,:,:]=image
        
        videos.append(frames)
        
    vidcap.release()
    cv2.destroyAllWindows()
    return videos


def list2Tensor(tensors):
    """
    tensors can be a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor):
        print("input is already a tensor")
        return tensors
            
        
    elif isinstance(tensors, (tuple, list)):
        assert type(tensors[0])==torch.Tensor, "input is not an iterable of torch tensors"
        #max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
        dimsizes=[tensors[0].size()[i] for i in range(tensors[0].dim())]
        dimsizes.insert(0, len(tensors))
        out=torch.zeros(dimsizes)
        for i in range(dimsizes[0]):
            out[i]=tensors[i]
        return out
            
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))

def _tie_weights(output_embeddings, input_embeddings):
        """ Tie input and output module weights
        """
        output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
            
def fill_tensor_1D(T:torch.Tensor, i:int, val):
    T[i]=val
    return T