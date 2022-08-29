import os
import collections
import sys

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

trainset = DecExpDataset('train')

def incremental_PCA(nr_of_samples,nr_components):

        train_loader = DataLoader(
                trainset, batch_size=3000,
                shuffle=True, num_workers=args.num_workers,
                drop_last=False, pin_memory=False)

        pca_res=np.zeros((1,nr_components))
        ipca = IncrementalPCA(n_components=nr_components)
        for x, y in tqdm(loader):
            #Flatten into 1D vector of features for PCA
            x=x.flatten().view(nr_of_samples,1572864).numpy()
            ipca.partial_fit(x)
        loader=self.create_loader(nr_of_samples)
        for x, y in tqdm(loader):
            x=x.flatten().view(nr_of_samples,1572864).numpy()
            tr=ipca.transform(x)
            pca_res=np.vstack((pca_res,tr))
        pca_array=np.array(pca_res)
        components=ipca.components_
        return pca_array[1:,:],components





barloader=tqdm(train_loader, total=self.batches_per_epoch)