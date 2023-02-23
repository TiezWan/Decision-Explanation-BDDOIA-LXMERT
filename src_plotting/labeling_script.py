import matplotlib.pyplot as plt
import matplotlib.image as mplim
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import os
import pdb
import json
import random
import shutil


root = tk.Tk()

root.geometry("300x300+2000+2000")

root.withdraw()


# # get screen width and height
# ws = root.winfo_screenwidth() # width of the screen
# hs = root.winfo_screenheight() # height of the screen

# # calculate x and y coordinates for the Tk root window
# x = (ws/2) - (w/2)
# y = (hs/2) - (h/2)

# # set the dimensions of the screen 
# # and where it is placed
# root.geometry('%dx%d+%d+%d' % (w, h, x, y))


split="test"
folders_path=f"{split}clips_downsampled_6fpv_2sampled/"
list_folders=os.listdir(folders_path)
label_dict={}
question_nr=1 #from 1 to n, not starting from 0, 2 by 2
savefile=f'bdd100k_subtasks_{split}labels.json'

def save():
    with open(savefile, 'w') as f:
        json.dump(label_dict, f)

# #Sampling 2 random images from the 6fpv folder
# for folder in list_folders:
#     os.makedirs("valclips_downsampled_6fpv_2sampled/"+folder)
#     choices=random.choices([i for i in range(len(os.listdir(folders_path+folder+"/"))-1)], k=2)
#     for choice in choices:
#         shutil.copyfile(folders_path+folder+"/"+folder+f"_frame_{choice}.jpeg", "valclips_downsampled_6fpv_2sampled/"+folder+"/"+folder+f"_frame_{choice}.jpeg")
# pdb.set_trace()


if os.path.isfile(savefile):
    label_dict=json.load(open(savefile, "r"))
    print("label_dict successfully loaded")
else:
    #Creating the labels dict 
    for folder in list_folders:
        temp={}
        for item in os.listdir(folders_path+folder+"/"):
            if item.split(".")[-1]=="jpg" or item.split(".")[-1]=="jpeg":
                temp[item]=[None for i in range(8)]
        label_dict[folder]=temp
pdb.set_trace()
count=0
### Autocomplete the inverse/anti- questions

for folder in list_folders:
    for item in os.listdir(folders_path+folder+"/"):
            if label_dict[folder][item][0]!=None:
                label_dict[folder][item][1]=1-label_dict[folder][item][0]
            if label_dict[folder][item][2]!=None:
                label_dict[folder][item][3]=1-label_dict[folder][item][2]
            if label_dict[folder][item][4]!=None:
                label_dict[folder][item][5]=1-label_dict[folder][item][4]
            if label_dict[folder][item][6]!=None:
                label_dict[folder][item][7]=1-label_dict[folder][item][6]
pdb.set_trace()
save()

nb_folders=len(list_folders)
curr_fold_nr=0

while curr_fold_nr<nb_folders:
    folder=list_folders[curr_fold_nr]
    item_nr=0
    while item_nr<len(os.listdir(folders_path+folder+"/")):
        skipflag=0
        item=os.listdir(folders_path+folder+"/")[item_nr]
        if label_dict[folder][item][question_nr-1]!=None:
            count+=1
            item_nr+=1
            skipflag=1
        if skipflag==0:
            if item.split(".")[-1]=="jpg" or item.split(".")[-1]=="jpeg":
                label=None
                while not ((label=="0") or (label=="1") or (label=="y") or (label=="n") or (label=="r")):
                    image = mplim.imread(folders_path+folder+"/"+item)
                    plt.imshow(image)
                    plt.ion()
                    plt.show()
                    label = simpledialog.askstring(title="Label",
                                                    prompt="Is there a green light in the image ? (1/y/0/n):")
                    plt.clf()
                    if label=="exit":
                        savevar=simpledialog.askstring(title="Save",
                                                    prompt="Save ?:")
                        if savevar=="y" or savevar=="1":
                            save()
                        raise Exception
                if label=="r":
                    item_nr-=1
                    if item_nr==-1:
                        curr_fold_nr-=1
                        item_nr=len(os.listdir(folders_path+list_folders[curr_fold_nr]+"/"))-1
                    if curr_fold_nr<0:
                        curr_fold_nr=0
                    folder=list_folders[curr_fold_nr]
                    item=os.listdir(folders_path+folder+"/")[item_nr]
                    label_dict[folder][item][question_nr-1]=None
                    count-=1
                if label=="y":
                    label=1
                elif label=="n":
                    label=0
                elif label=="1":
                    label=1
                elif label=="0":
                    label=0
                if label==1 or label==0:
                    label_dict[folder][item][question_nr-1]=label
                    count+=1
                    item_nr+=1
                    #print(count)
                del image
        print(count)
    curr_fold_nr+=1
save()
        
root.mainloop()