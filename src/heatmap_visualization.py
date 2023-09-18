import numpy as np
import torch
from torch import Tensor
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from typing import List, Tuple, Union, Dict
import os
import json
from sklearn.cluster import DBSCAN
from src.utils.param import args

matplotlib.rcParams.update({'font.size': 7})

class HeatmapVisualization:
    def __init__(self, num_heads=12, num_objs=100, available_questions=[0, 1, 2], pooling="sum", input_path=None, img_save_path=None, heatmap_labels=None) -> None:
        self.num_heads = num_heads
        self.num_objs = num_objs
        # questions_class[0] is red light type question; questions_class[1] is green lgiht question; questions_class[2] is road sign type question 
        self.available_questions = available_questions 
        self.pooling = pooling
        self.img_save_path = img_save_path
        self.input_path = input_path
        self.orig_frame_path = os.path.join(input_path, 'testclips_heatmap_images')
        self.feature_path = os.path.join(input_path, 'feature_output', 'testclips_downsampled_6fpv_2sampled')
        self.heatmap_labels = heatmap_labels
    
    # load one of labeled 3 types features: redlights, greenlights, and road signs of each image
    def load_labels(self, ques_nr: int) -> Dict:
        one_of_labels = {}
        # each type label is saved in a json file
        path = self.heatmap_labels[ques_nr]
        file_path = f'{self.input_path}/{path}'
        with open(file_path, 'r') as f:
            labels = json.load(f)
        for key in labels:
            for frame in labels[key]:
                one_of_labels[frame] = labels[key][frame]
        return one_of_labels

    # according to attention scores to marked the interest objects 
    def compute_objs(self,imgid: str, batch_attention_scores: Tensor, sent: str, ques_nr: int, cluster_lenth: int, eps_dbscan: float, plot=True,) -> List:
        # Check pytoch source code as a ref to write docstring
        # attention_scores: pytorch [12, 15, 100]
        # eps_dbscan: clustering parameters
        # cluster_lenth:If the cluster of highest value objects has a length bigger than this, plot nothing
        # This choice is justified by the working of DBSCAN: if the cluster contains a lot of
        # objects, it is likely that there are lots of objects unrelated to the question

        # Maximum number of bbox to plot in clustering mode
        bboxcaplength=None 
        interestobjs = []       
        queries = sent.split(" ")
        num_words = len(queries)
        # creating a copy that will then be sent to cpu, to avoid sending original to cpu        
        attention_scores = batch_attention_scores.clone()
        # dividing the square of heads 
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.num_heads, dtype=torch.float32))
        # Calculate the sum of the absolute values of all heads' attention_scores 
        #attention_scores[attention_scores < 0] = 0
        scores_allheads_sum = torch.sum(abs(attention_scores), dim=0)
        #scores_allheads_norm = scores_allheads_sum / torch.max(scores_allheads_sum)
        # Obtain the attention probabilities by softmax function           
        source = scores_allheads_sum.squeeze(0).detach().cpu().numpy()
        normal_source = (source - source.min()) / (source.max() - source.min())
        # Switch the pooling way
        if self.pooling == "CLS":
            max_value = np.amax(normal_source[0])
            np_source = np.array(normal_source[0] / max(normal_source[0])) # CLS is the first row of matrix
        elif self.pooling == "sum":
            np_source = np.sum(normal_source[1:(num_words+1), :], 0) # Average the rows of matrix
            max_value = np.amax(np_source)
            np_source /= max_value #Normalizing 
            #print(np.shape(source)): (15,100)
        # Using DESCAN to find interest objects
        np_source_ex1 = np.expand_dims(np_source, 1)
        cluster_algo = DBSCAN(eps=eps_dbscan, min_samples=1)
        cluster_labels = cluster_algo.fit_predict(np_source_ex1)
        cluster_centers = [[] for i in range(max(cluster_labels)+1)]
        for objnr, objcluster in enumerate(cluster_labels):
            cluster_centers[objcluster].append(np_source_ex1[objnr,0])
        for cluster_nr in range(len(cluster_centers)):
            cluster_centers[cluster_nr] = np.mean(cluster_centers[cluster_nr])

        cluster_max_idx = np.argmax(cluster_centers)
        interestobjs=[idx for idx in range(len(cluster_labels)) if cluster_labels[idx] == cluster_max_idx]
        if len(interestobjs)>cluster_lenth:
            interestobjs=[]
        if bboxcaplength!=None:
            if len(interestobjs)>bboxcaplength:
                interestobjs=np.flip(np.argsort(np_source_ex1, 0))[:,0][:bboxcaplength] #Get the bboxcaplength with highest values
        if plot and interestobjs != []:
            self.plot_heatmap(imgid, normal_source, ques_nr)
            self.plot_bboxes(imgid, np_source, interestobjs, ques_nr)
                        
        return interestobjs

    def evaluate(self, listimgids, num_interest_objs, ques_nr: int) -> Tuple:
        #interestobjs is a numpy array of size (3, num_img, num_objs) 
        true_pos=0
        false_pos=0
        false_neg=0
        true_neg=0

        labels = self.load_labels(ques_nr)
        for img_nr, imgid in enumerate(listimgids):
            none_interestobjs = all(x == -1 for x in num_interest_objs[img_nr])
            if none_interestobjs:
                break
            else:
                for obj in range(self.num_objs):
                    if (obj in num_interest_objs[img_nr]) and (obj in labels[imgid]):
                        true_pos += 1
                    elif (obj in num_interest_objs[img_nr]) and not (obj in labels[imgid]):
                        false_pos += 1 
                    elif not (obj in num_interest_objs[img_nr]) and (obj in labels[imgid]):
                        false_neg += 1
                    elif not (obj in num_interest_objs[img_nr]) and not (obj in labels[imgid]):
                        true_neg += 1       
        return true_pos, false_pos, false_neg, true_neg
    
    def pad_list(self, list_to_pad: List, padlength: int, token) -> List:
        # pad tokens into list to uniform length
        if isinstance(list_to_pad, np.ndarray):
                list_to_pad = list_to_pad.ravel().tolist()
        elif not list_to_pad:  # Use this condition to check for an empty list
            list_to_pad = padlength * [token]
        while len(list_to_pad) < padlength:
            list_to_pad.append(token)
        return list_to_pad

    def plot_bboxes(self, imgid, np_source, roi_idx, ques_nr: int) -> None:
        # Plotting the bounding boxes on the screen
        # load original frame
        img_path = os.path.join(self.orig_frame_path, "_".join(imgid.split("_")[:3]), imgid + ".jpg")
        if os.path.exists(img_path):
            imgplot = mpimg.imread(img_path)
        else:
            print(f"Image file does not exist: {img_path}")

        features_path = os.path.join(self.feature_path, imgid + ".npy")
        if os.path.exists(features_path):
            features = np.load(features_path, allow_pickle=True)[0]
        else:
            print(f"Features file does not exist: {features_path}")

        # obtain bounding boxes
        bboxes = features['bbox']           
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        bbox_source = [value if i in roi_idx else 0 for i, value in enumerate(np_source)]
        img1 = ax1.imshow(imgplot, cmap='jet', aspect='auto')
        for i in roi_idx:
            bbox = bboxes[i]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1,
                                    edgecolor='r', facecolor='none')
            ax1.add_patch(rect)

        ax2.grid(True)
        bar_width = 1
        x_positions = range(len(np_source))
        bar_colors = ['green' if value != 0 else 'gray' for value in bbox_source]
        ax2.bar(x_positions, np_source, bar_width, color=bar_colors, edgecolor='white')
        ax2.set_ylim(0, max(np_source))       
        legend_handles = [patches.Patch(color='green', label= 'Objects of interest')]
        ax2.legend(handles=legend_handles, loc='upper right', facecolor='white', edgecolor='black')       
        save_path = os.path.join(self.img_save_path, f"{imgid}_sent{ques_nr}_bboxes.jpg")
        plt.savefig(save_path)
        # plt.show()

            
    def plot_heatmap(self, imgid, source, ques_nr: int) -> None:
        # plot heatmap of attention scores
        fig, ax = plt.subplots(figsize = (16,10))
        img = ax.imshow(source, cmap='jet')
        x_label = np.arange(0, 100, 10)
        y_label = np.arange(0, 15, 5)
        ax.set_xticks(x_label)
        ax.set_yticks(y_label)
        ax.set_xticklabels(x_label)
        ax.set_yticklabels(y_label)
        ax.set_xlabel('Objects')
        ax.set_ylabel('Words')
        bar = plt.colorbar(img, ax = ax, shrink = 0.2, pad = 0.02)
        plt.savefig(f'{self.img_save_path}/{imgid}_sent{ques_nr}_heatmap.jpg')
        #plt.show()

    def run_and_eval(self, listimgids: List, attention_scores: Tensor, ques_nr: int, sents: Tuple, cluster_lenth: int, eps_dbscan: float) -> Tuple:
        # the all number of objects of interest in image list
        num_interest_objs = np.zeros([len(listimgids), self.num_objs])
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0

        if sents is not None:
            for img_nr, img_id in enumerate(listimgids):
                # compute the objects of interest from attention scores
                interest_objs = self.compute_objs(img_id, attention_scores[img_nr,:,:,:], sents[img_nr], ques_nr, cluster_lenth, eps_dbscan, plot=args.plot_heatmap)
                interest_temp = self.pad_list(np.array(interest_objs), self.num_objs, -1) #Padding with -1 up to number of all objects
                num_interest_objs[img_nr] = np.array(interest_temp)
                
            true_pos, false_pos, false_neg, true_neg = self.evaluate(listimgids, num_interest_objs, ques_nr)
            return (true_neg, false_pos, false_neg, true_pos)


