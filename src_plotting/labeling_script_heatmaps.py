import matplotlib.pyplot as plt
import matplotlib.image as mplim
import matplotlib.patches as patches
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import os
import pdb
import json
import random
import shutil
from matplotlib.widgets import Slider, Button, TextBox


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
images_path=f"C:\\Users\\Wantiez\\Documents\\FP\\bdd100k_heatmaps\\12_9_5_5_bs\
8_mixed_tasks_5em5_corrected_pooler_shuffle_epoch_19\\allsents_per_sample\\seed_158467\\{split}_set\\"
list_images=os.listdir(images_path)

##Removing duplicates
# skipflag=0
# for i in range(len(list_images)):
#     if skipflag==0:
#         try:
#             if list_images[i].split("_")[:-2]==list_images[i+1].split("_")[:-2]:
#                 skipflag=1
#                 list_images_no_dupe.append("_".join(list_images[i].split("_")[:-2]))
#             else:
#                 list_images_no_dupe.append("_".join(list_images[i].split("_")[:-2]))
#         except IndexError as e:
#             print(e)
#             pass
#     else:
#         skipflag=0
#         continue

image_path=f"{split}clips_downsampled_6fpv_2sampled\\"
label_dict={}
question_nr=1 #from 1 to 3 (redlights, greenlights, roadsigns)
if question_nr==1:
    savefile=f'bdd100k_heatmaps_{split}labels_redlights.json'
elif question_nr==2:
    savefile=f'bdd100k_heatmaps_{split}labels_greenlights.json'
elif question_nr==3:
    savefile=f'bdd100k_heatmaps_{split}labels_roadsigns.json'

def save():
    with open(savefile, 'w') as f:
        json.dump(label_dict, f)
    print("Successfully saved")

# #Sampling 2 random images from the 6fpv folder
# for folder in list_images:
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
    for image in list_images:
        imgfolder="_".join(image.split("_")[:-2])
        try:
            label_dict[imgfolder].update({image:[None]})
        except KeyError:
            label_dict[imgfolder]={image:[None]}
            
count=0

nb_folders=len(list_images)
curr_fold_nr=0
label_dict_folders=list(label_dict.keys())

def flip_bbox_around_center(bbox, width, height):
    for i, point in enumerate(bbox):
        if i%2==0:
            bbox[i]=width-bbox[i]
        else:
            bbox[i]=height-bbox[i]
    return bbox

while curr_fold_nr<nb_folders:
    try:
        folder=label_dict_folders[curr_fold_nr]
        item_nr=0
        while item_nr<len(label_dict[folder]):
            skipflag=0
            keysforframes=list(label_dict[folder].keys())
            item=keysforframes[item_nr]
            if label_dict[folder][item][0]!=None:
                count+=1
                item_nr+=1
                skipflag=0
            if skipflag==0:
                label=None
                latch=True
                while latch==True:
                    image = mplim.imread(image_path+folder+"/"+item+".jpg")
                        
                    #image = mplim.imread(image_path+"1a3dc8be-62d054ea_34_40/1a3dc8be-62d054ea_34_40_frame_1.jpg")
                    #A=np.load(f"C:/Users/Wantiez/Desktop/1a3dc8be-62d054ea_34_40_frame_1.npy", allow_pickle=True)[0]
                    
                    A=np.load(f"./bdd100k/feature_output/testclips_downsampled_6fpv_2sampled/"+item+".npy", allow_pickle=True)[0] #Image to load
                    #A=np.load(f"./bdd100k/feature_output/{split}/{split}clips_downsampled_6fpv/"+folder+".npy", allow_pickle=True)[int(item.split("_")[-1].split(".")[0])] #Image to load
                    boxes_unscaled=A['bbox']
                    #Flip all bounding boxes
                    for bbox_nr, bbox in enumerate(boxes_unscaled):
                        #boxes_unscaled[bbox_nr]=flip_bbox_around_center(bbox, img_w, img_h)
                        boxes_unscaled[bbox_nr]=bbox
                    curr_box=0
                    ax1=plt.subplot(111)
                    bbox=boxes_unscaled[curr_box]
                    ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none'))
                    ax1.text(bbox[0],bbox[3], str(curr_box))# + ", " + obj_cats[A['objects'][i]]['name'])
                    ax1.imshow(image)
                    
                    axlbd = plt.axes([0.1, 0.15, 0.0225, 0.8])
                    
                    #Buttons
                    axup = plt.axes([0.65, 0.025, 0.1, 0.04])
                    axdown = plt.axes([0.5, 0.025, 0.1, 0.04])
                    #axok = plt.axes([0.8, 0.025, 0.1, 0.04])
                    upbutton = Button(axup, '+', hovercolor='0.975')
                    downbutton = Button(axdown, '-', hovercolor='0.975')
                    #okbutton = Button(axok, 'Next', hovercolor='0.975')
                    
                    # #Input fields
                    # axboxred = plt.axes([0.13, 0.94, 0.8, 0.05])
                    # text_box_red = TextBox(axboxred, 'Redlights', initial="")
                    # axboxgreen = plt.axes([0.13, 0.87, 0.8, 0.05])
                    # text_box_green = TextBox(axboxgreen, 'Greenlights', initial="")
                    # axboxroad = plt.axes([0.13, 0.8, 0.8, 0.05])
                    # text_box_road = TextBox(axboxroad, 'Roadsigns', initial="")

                    lmbd_slider = Slider(
                        ax=axlbd,
                        label='nb_boxes',
                        valmin=0,
                        valmax=99,
                        valinit=0,
                        valstep=1,
                        orientation='vertical'
                    )
                    
                    # The function to be called anytime a slider's value changes
                    def update(val):
                        global curr_box
                        curr_box=val
                        ax1.clear()
                        bbox=boxes_unscaled[val]
                        
                        ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='r', facecolor='none'))
                        ax1.add_patch(patches.Rectangle((bbox[2], bbox[3]), -35, 20, linewidth=1, edgecolor='r', facecolor='r'))
                        ax1.text(bbox[2]-23,bbox[3]+15, str(val))# + ", " + obj_cats[A['objects'][i]]['name'])
                        ax1.imshow(image)
                        
                        #ax1.canvas.draw_idle()
                    
                    def inc_bbox(event):
                        global curr_box
                        curr_box+=1
                        update(curr_box)
                    
                    def dec_bbox(event):
                        global curr_box
                        curr_box-=1
                        update(curr_box)
                        
                    def submit(text):
                        label_field=text
                        print(label_field)
                        
                    plt.ion()
                    lmbd_slider.on_changed(update)
                    curr_box = upbutton.on_clicked(inc_bbox)
                    curr_box = downbutton.on_clicked(dec_bbox)
                    #text_box_red.on_submit(submit)
                    #text_box_green.on_submit(submit)
                    #text_box_road.on_submit(submit)
                    
                    
                    plt.show()
                    label = simpledialog.askstring(title="Label",
                                                    prompt="Select the ids of relevant bounding boxes:")
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
                            item_nr=len(label_dict[label_dict_folders[curr_fold_nr]])-1
                        if curr_fold_nr<0:
                            curr_fold_nr=0
                        folder=label_dict_folders[curr_fold_nr]
                        keysforframes=list(label_dict[folder].keys())
                        item=keysforframes[item_nr]
                        label_dict[folder][item]=[None]
                        count-=1
                        if count==-1:
                            count=0
                        latch=False
                    
                    elif label=="": #If no bounding box are interesting
                        count+=1
                        item_nr+=1
                        label_dict[folder][item]=["None"]
                        latch=False
                    
                    else:
                        label=label.strip()
                        try:
                            labels=[int(label.split(" ")[i]) for i in range(len(label.split(" ")))]
                            label_dict[folder][item]=labels
                            count+=1
                            item_nr+=1
                            latch=False
                            #print(count)
                        
                        except ValueError: #If the label is incorrect
                            pdb.set_trace()
                            pass
                    
                        except AttributeError: #When pressing the cancel button
                            savevar=simpledialog.askstring(title="Save",
                                                    prompt="Save ?:")
                            if savevar=="y" or savevar=="1":
                                save()
                            raise Exception
                    del image
                        
            print("count :", count)
        curr_fold_nr+=1
    except IndexError as e:
        print(e)
        savevar=simpledialog.askstring(title="Save", prompt="Save ?:")
        if savevar=="y" or savevar=="1":
            save()
        raise Exception
        
save()

root.mainloop()