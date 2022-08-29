import os
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

class ISVQADataSet(Dataset):
    def __init__(self, annotations_files, img_dir, transform=None, target_transform=None):
        self.data = json.load(open(annotations_files))['data']
        intermediate_path = os.listdir(img_dir)
        self.img_dir = [img_dir + '/' + p + '/samples' for p in intermediate_path if p[:4]=='part']
        print(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx]['answers']
        question = self.data[idx]['question_str']
        image_names = self.data[idx]['image_names']
        image_set = []

        for i in range(len(image_names)):
            for j, img_dir in enumerate(self.img_dir):
                img_path = os.path.join(img_dir, image_names[i]+'.jpg')
                if os.path.exists(img_path):
                    image = read_image(img_path)
                    if self.transform:
                        image = self.transform(image)
                    image_set.append(image)
                    break
                elif j == len(self.img_dir)-1:
                    question = None
                    label = None

        if self.target_transform:
            label = self.target_transform(label)

        return image_set, question, label

batch_size = 1
dataset = ISVQADataSet(
    './input/ISVQA/Annotation/imdb_nuscenes_trainval.json', 
    img_dir='input/ISVQA/NuScenes'
    )
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

for n, (ims, q, l) in enumerate(dataloader):
    if q is not None and l is not None:
        print(n)
        print(q, l)
        for k, im in enumerate(ims):
            im = np.squeeze(im, axis=0)
            print(im.shape)
            plt.subplot(321+k)
            plt.imshow(im.permute(1, 2, 0))
        plt.savefig('./output/{}.jpg'.format(n))
        plt.close()
        break

     
    # for i, im in enumerate(x[j]):
    #     plt.subplot(321+i)
    #     plt.imshow(im)
    # plt.savefig('./output/{}.jpg'.format(j))
    # plt.close()
