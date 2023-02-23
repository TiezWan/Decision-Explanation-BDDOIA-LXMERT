import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import os
import json
from matplotlib.widgets import Slider, Button
import random
import pdb
from sklearn.cluster import DBSCAN

matplotlib.rcParams.update({'font.size': 7})

#heatmap_lang=np.load("abcde.npy")
nbheads=12
#queries=["[CLS]","traffic", "light", "red", "yellow", "green", "lane", "straight", "left", "right", "obstacle", "car", "person", "rider", "other", "sign", "stop", "[EOS]"]
#queries=["[CLS]", "person", "truck", "headlight", "sky", "license", "plate", "bicycle", "windshield", "people", "child", "[SEP]"]
#queries=["[CLS]","table", "guitar", "tobacco", "shelf", "pen", "screw", "plate", "keyboard", "laser", "magnet", "[SEP]"]
#queries=["[CLS]", "building", "building", "flag", "flag", "flag", "flag", "building", "light", "light", "street", "people", "light", "street", "building", "building", "flag", "street", "car", "man", "window", "light", "window", "light", "people", "crosswalk", "street", "building", "sign", "light", "window", "building", "building", "man", "window", "sign", "window", "building", "street", "sign", "man", "flag", "flag", "window", "building", "window", "person", "man", "window", "taxi", "street", "light", "light", "street", "window", "building", "light", "light", "street", "flag", "window", "line", "sign", "street", "window", "windows", "street", "window", "light", "car", "man", "car", "light", "building", "city", "window", "sign", "pole", "sign", "street", "window", "light", "window", "light", "crosswalk", "window", "awning", "person", "building", "light", "car", "window", "people", "building", "street", "building", "flag", "crosswalk", "street", "window", "traffic", "[SEP]"] 
#queries=["[CLS]", "building", "[SEP]"]


attsc_layer_1_lang_currimg=[]
attsc_layer_2_lang_currimg=[]
attsc_layer_3_lang_currimg=[]
attsc_layer_4_lang_currimg=[]
attsc_layer_5_lang_currimg=[]
img_to_plot='1f2ccb3c-9226290d_0_40_frame_1'

q_nr=str(2)
ctx=0
att=1
lang=1
visn=0
mode='x'
pooling="sum"

frame_nr=int(img_to_plot.split("_")[-1])
imgdir=f"12_9_5_5_bs8_mixed_tasks_5em5_corrected_pooler_shuffle_epoch_19/allsents_per_sample/seed_158467/test_set"

# 12_9_5_5_bs8_mixed_tasks_epoch_20
# 12_9_5_5_bs8_mixed_tasks_5em5_corrected_pooler_shuffle_epoch_19
# 12_9_5_5_bs8_mixed_benchmark_lxmert_epoch_17

with open(f"./bdd100k_heatmaps/{imgdir}/{img_to_plot}/sent{q_nr}/sents.txt", 'r') as f:
    sent=f.read()
    queries=sent.split(" ")
    nbwords=len(queries)
print(queries)


#43f61d4d-25078e4b_3
for i in range(nbheads):
    #attsc_layer_1_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/attention_scores/lang/layer_1/"+img_to_plot+"_attsc_layer_1_lang_head_"+ str(i+1)+".npy"))

    #attsc_layer_2_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/attention_scores/lang/layer_2/"+img_to_plot+"_attsc_layer_2_lang_head_"+ str(i+1)+".npy"))
    
    #attsc_layer_3_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/attention_scores/lang/layer_3/"+img_to_plot+"_attsc_layer_3_lang_head_"+ str(i+1)+".npy"))

    #attsc_layer_4_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/attention_scores/lang/layer_4/"+img_to_plot+"_attsc_layer_4_lang_head_"+ str(i+1)+".npy"))

    attsc_layer_5_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
        + "/sent"+q_nr+"/attention_scores/lang/layer_5/"+img_to_plot+"_attsc_layer_5_lang_head_"+ str(i+1)+".npy"))


attsc_layer_1_visn_currimg=[]
attsc_layer_2_visn_currimg=[]
attsc_layer_3_visn_currimg=[]
attsc_layer_4_visn_currimg=[]
attsc_layer_5_visn_currimg=[]
#for i in range(nbheads):
    #attsc_layer_1_visn_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/attention_scores/visn/layer_1/"+img_to_plot+"_attsc_layer_1_visn_head_"+ str(i+1)+".npy"))

    #attsc_layer_2_visn_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/attention_scores/visn/layer_2/"+img_to_plot+"_attsc_layer_2_visn_head_"+ str(i+1)+".npy"))

    #attsc_layer_3_visn_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "attention_scores/visn/layer_3/"+img_to_plot+"_attsc_layer_3_visn_head_"+ str(i+1)+".npy"))

    #attsc_layer_5_visn_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/attention_scores/visn/layer_5/"+img_to_plot+"_attsc_layer_5_visn_head_"+ str(i+1)+".npy"))


ctxlay_layer_1_lang_currimg=[]
ctxlay_layer_2_lang_currimg=[]
ctxlay_layer_3_lang_currimg=[]
#for i in range(nbheads):
    #ctxlay_layer_1_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/context_layer/lang/layer_1/"+img_to_plot+"_ctxlay_layer_1_lang_head_"+ str(i+1)+".npy"))

    #ctxlay_layer_2_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/context_layer/lang/layer_2/"+img_to_plot+"_ctxlay_layer_2_lang_head_"+ str(i+1)+".npy"))

    #ctxlay_layer_3_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "context_layer/lang/layer_3/"+img_to_plot+"_ctxlay_layer_3_lang_head_"+ str(i+1)+".npy", allow_pickle=True))


ctxlay_layer_1_visn_currimg=[]
ctxlay_layer_2_visn_currimg=[]
ctxlay_layer_3_visn_currimg=[]
#for i in range(nbheads):
    #ctxlay_layer_1_visn_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/sent"+q_nr+"/context_layer/visn/layer_1/"+img_to_plot+"_ctxlay_layer_1_visn_head_"+ str(i+1)+".npy"))

    #ctxlay_layer_2_visn_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "/context_layer/visn/layer_2/"+img_to_plot+"_ctxlay_layer_2_visn_head_"+ str(i+1)+".npy"))

    #ctxlay_layer_3_visn_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+img_to_plot \
    #    + "context_layer/visn/layer_3/"+img_to_plot+"_ctxlay_layer_3_visn_head_"+ str(i+1)+".npy"))


def flip_bbox_around_center(bbox, width, height):
    for i, point in enumerate(bbox):
        if i%2==0:
            bbox[i]=width-bbox[i]
        else:
            bbox[i]=height-bbox[i]
    return bbox

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

#heatmap_visn=np.load('fghij.npy')
#print(np.shape(heatmap_lang[0,0,:,:]))
#print(np.shape(heatmap_visn[0,0,:,:]))
#plt.imshow(heatmap_lang[0,0,:,:], extent=[0, 100, 0, 50])
#plt.imshow(heatmap_visn[0,0,:,:], extent=[0, 50, 0, 100])
#shw=plt.imshow(attsccurrimg[0], extent=[0, 100, 0, 50], cmap='jet')


#sum_layer_3_lang=np.zeros([18,100])
#for i in range(12):
#    sum_layer_3_lang+=attsc_layer_2_lang_currimg[i]

#corrmat=np.zeros([18,1])
#for i in range(18):
#    corrmat[i]=np.linalg.norm(attsc_layer_2_lang_currimg[0][0] - attsc_layer_2_lang_currimg[0][i], 2)/476.61221313

#Trying source as the sum of normalized lang+visn attention_scores
#langattsc=abs(attsc_layer_2_lang_currimg[head_to_plot-1])
#visnattsc=abs(attsc_layer_2_visn_currimg[head_to_plot-1])
#source=((langattsc/np.amax(langattsc))+(visnattsc.T/np.amax(visnattsc)))

lambda_thr=0.8
nbboxes=5
plotmode="clustering"  #"nbboxes", "lambda_thr" or "clustering"
plotamplitudes=True #Plots amplitude of the source over the heatmaps

##Clustering parameters
eps_dbscan=0.128
bboxcaplength=None #Maximum number of bbox to plot in clustering mode
clusterlenthr=35 #If the cluster of highest value objects has a length bigger than this, plot nothing
                #This choice is justified by the working of DBSCAN: if the cluster contains a lot of
                #objects, it is likely that there are lots of objects unrelated to the question

# fig, (ax1, ax2) = plt.subplots(2,1)
# #img = ax.imshow(sum([abs(attsc_layer_2_lang_currimg[i]) for i in range(nbheads)]), extent=[0, 100, 0, 54], cmap='jet')
# #img = ax.imshow(abs(sum_layer_3_lang), extent=[0, 100, 0, 54], cmap='jet')
# img = ax2.imshow(source, extent=[0, 100, 0, 54], cmap='jet')


#Plotting attention_scores
if att==1:
    if lang==1:
        head_to_plot=4
        source=sum([abs(attsc_layer_5_lang_currimg[i]) for i in range(nbheads)])
        #source=abs(attsc_layer_5_lang_currimg[head_to_plot-1])
        #print(np.shape(source))
        
        if mode=='x':
            if pooling=="CLS":
                maxvalue=np.amax(source[0])
            elif pooling=="sum":
                maxvalue=np.amax(np.sum(source[1:(nbwords+1),:], 0))
        
        #Deprecated
        if mode=='lr':
            source=source[1:-1]
            maxvalue=np.amax(source)

        fig, (ax1, ax2) = plt.subplots(2,1)
        #img = ax.imshow(sum([abs(attsc_layer_2_lang_currimg[i]) for i in range(nbheads)]), extent=[0, 100, 0, 54], cmap='jet')
        #img = ax.imshow(abs(sum_layer_3_lang), extent=[0, 100, 0, 54], cmap='jet')
        
        if pooling=="CLS":
            np_source = np.array(source[0]/max(source[0]))
            
        elif pooling=="sum":
            np_source = np.sum(source[1:(nbwords+1),:],0)
            np_source /= max(np_source) #Normalizing
            
        #img = ax2.imshow(source, extent=[0, 100, 0, 45], cmap='jet')
        img = ax2.imshow(np.expand_dims(np_source,1).T, extent=[0, 100, 0, 45], cmap='jet')
        
        if plotamplitudes:
            ax3=ax2.twinx()
            ax3.set_ylim(0, max(np_source)*1.1)
            ax3.scatter([0 for i in range(100)], np_source)
            for obj in range(np_source.shape[0]-1):
                ax3.add_patch(patches.Rectangle((obj, np_source[obj]), 1, 0.001, edgecolor='white'))
                ax3.add_patch(patches.Rectangle((obj+0.995, np_source[obj]), 0.001, np_source[obj+1]-np_source[obj], edgecolor='white'))
        
        y_label_list = [15,10,5,1]
        ax2.set_yticks([1.5,16.5,31.5,43.5]) #Positions on the y axis for ticks in y_label_list. Refer to img to know height bounds
        ax2.set_yticklabels(y_label_list)
        bar = fig.colorbar(img, ax=ax2)
        strqueries="[CLS]\n"
        for i in queries[(1*mode=="lr"):nbwords-(1*mode=='lr')]: #Get rid of [CLS] and [SEP] labels on axis for 'lr' mode
            strqueries+=i
            strqueries+="\n"
        strqueries+="[SEP]\n"
        for i in range(15-len(strqueries.split("\n"))+1):
            strqueries+="[PAD]\n"
        ax2.set_ylabel(strqueries, rotation='horizontal')
        ax2.yaxis.set_label_coords(-0.12,-0.06)

        os.chdir("C:/Users/wantiez/Documents/FP")
        
        imgplot=mpimg.imread("./testclips_downsampled_6fpv_2sampled/"+"_".join(img_to_plot.split("_")[:3])+"/"+img_to_plot+".jpg")
        img_w=imgplot.shape[1] #xmax
        img_h=imgplot.shape[0] #ymax
        
        A=np.load("./bdd100k/feature_output/testclips_downsampled_6fpv_2sampled/"+img_to_plot+".npy", allow_pickle=True)[0] #Image to load
        
        boxes_unscaled=A['bbox']
        #Flip all bounding boxes
        for bbox_nr, bbox in enumerate(boxes_unscaled):
            #boxes_unscaled[bbox_nr]=flip_bbox_around_center(bbox, img_w, img_h)
            boxes_unscaled[bbox_nr]=bbox
        
        interestobjs=[]

        if mode=='x':
            if pooling=="CLS":
                #lambdathr plot mode
                if plotmode=="lambda_thr":
                    for obj in range(len(source[0])):
                        if source[0][obj]>lambda_thr*maxvalue:
                            interestobjs.append(obj)
                
                #nbboxes plot mode
                elif plotmode=="nbboxes":
                    np_source = np.array(source[0])
                    sorted_objs= np.sort(np_source)
                    interestobjs = np.flip(np.argsort(np_source))[:nbboxes]
                   
                #clustering plot mode
                elif plotmode=="clustering":
                    np_source=np.expand_dims((source[0]/max(source[0])), 1)
                    cluster_algo=DBSCAN(eps=eps_dbscan, min_samples=1)
                    cluster_data=cluster_algo.fit_predict(np_source)
                    cluster_centers=[[] for i in range(max(cluster_data)+1)]
                    for objnr, objcluster in enumerate(cluster_data):
                        cluster_centers[objcluster].append(np_source[objnr,0])
                    for clusternr in range(len(cluster_centers)):
                        cluster_centers[clusternr]=np.mean(cluster_centers[clusternr])
                    #print("cluster labels :", cluster_data)
                    #print("cluster centers :", cluster_centers)
                    cluster_to_plot=np.argmax(cluster_centers)
                    interestobjs=[idx for idx in range(len(cluster_data)) if cluster_data[idx]==cluster_to_plot]
                    if len(interestobjs)>clusterlenthr:
                        interestobjs=[]
                    if bboxcaplength!=None:
                        if len(interestobjs)>bboxcaplength:
                            interestobjs=np.flip(np.argsort(np_source, 0))[:,0][:bboxcaplength] #Get the bboxcaplength with highest values
                         
                            
            elif pooling=="sum":
                #lambdathr plot mode
                if plotmode=="lambda_thr":
                    for obj in range(len(source[0])):
                        if np.sum(source[1:(nbwords+1),:],0)[obj]>lambda_thr*maxvalue:
                            interestobjs.append(obj)
                
                #nbboxes plot mode
                elif plotmode=="nbboxes":
                    np_source = np.sum(source[1:(nbwords+1),:],0)
                    sorted_objs= np.sort(np_source)
                    interestobjs = np.flip(np.argsort(np_source))[:nbboxes]

                #clustering plot mode 
                elif plotmode=="clustering":
                    np_source = np.sum(source[1:(nbwords+1),:],0)
                    np_source = np.expand_dims((np_source/max(np_source)), 1)
                    cluster_algo=DBSCAN(eps=eps_dbscan, min_samples=1)
                    cluster_data=cluster_algo.fit_predict(np_source)
                    cluster_centers=[[] for i in range(max(cluster_data)+1)]
                    for objnr, objcluster in enumerate(cluster_data):
                        cluster_centers[objcluster].append(np_source[objnr,0])
                    for clusternr in range(len(cluster_centers)):
                        cluster_centers[clusternr]=np.mean(cluster_centers[clusternr])
                    #print("cluster labels :", cluster_data)
                    #print("cluster centers :", cluster_centers)
                    cluster_to_plot=np.argmax(cluster_centers)
                    interestobjs=[idx for idx in range(len(cluster_data)) if cluster_data[idx]==cluster_to_plot]
                    if len(interestobjs)>clusterlenthr:
                        interestobjs=[]
                    if bboxcaplength!=None:
                        if len(interestobjs)>bboxcaplength:
                            interestobjs=np.flip(np.argsort(np_source, 0))[:,0][:bboxcaplength] #Get the bboxcaplength with highest values
                    
        ##Deprecated for now
        if mode=='lr':
            for query in range(len(source)):
                for obj in range(len(source[0])):
                    if source[query][obj]>lambda_thr*maxvalue:
                        interestobjs.append((query,obj))
        
        obj_cats=list(json.load(open("visual_genome_categories.json")).values())[0]
        
        im2=ax1.imshow(imgplot)
        
        ### Plotting the bounding boxes on the screen
        for i in interestobjs:
            if mode=='x':
                bbox=boxes_unscaled[i]
            if mode=='lr':
                bbox=boxes_unscaled[i[1]]
            ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none'))
            #ax1.add_patch(patches.Rectangle((bbox[2], bbox[3]), -55, 40, linewidth=1, edgecolor='r', facecolor='r'))
            #ax1.text(bbox[2]-55,bbox[3]+37, str(i))# + ", " + obj_cats[A['objects'][i]]['name'])

            #Deprecated
            if mode=='lr':
                ax1.text(bbox[0],bbox[3], str(i))#+", q: "+str(queries[i[0]])+", cat: "+obj_cats[A['objects'][i[1]]]['name'])

        #Creating the slider to vary important parameters
        axlbd = plt.axes([0.1, 0.15, 0.0225, 0.8])
        if plotmode=="lambda_thr":
            lmbd_slider = Slider(
                ax=axlbd,
                label='lambda_thr',
                valmin=0,
                valmax=1,
                valinit=lambda_thr,
                orientation='vertical'
            )

        elif plotmode=="nbboxes":
            lmbd_slider = Slider(
                ax=axlbd,
                label='nb_boxes',
                valmin=0,
                valmax=20,
                valinit=nbboxes,
                valstep=1,
                orientation='vertical'
            )
            
        elif plotmode=="clustering":
            lmbd_slider = Slider(
                ax=axlbd,
                label='epsilon',
                valmin=0,
                valmax=0.2,
                valinit=eps_dbscan,
                valstep=0.01,
                orientation='vertical'
            )

        # The function to be called anytime a slider's value changes
        def update(val):
            ax1.clear()
            interestobjs=[]
            if val==0:
                val=0.00001

            if mode=='x':
                if pooling=="CLS":
                    #lambdathr plot mode
                    if plotmode=="lambda_thr":
                        for obj in range(len(source[0])):
                            if source[0][obj]>val*maxvalue:
                                interestobjs.append(obj)
                            
                    #nbboxes plot mode
                    elif plotmode=="nbboxes":
                        np_source = np.array(source[0])
                        sorted_objs= np.sort(np_source)
                        interestobjs = np.flip(np.argsort(np_source))[:val]
                        
                    #clustering plot mode
                    elif plotmode=="clustering":
                        np_source=np.expand_dims((source[0]/max(source[0])), 1)
                        cluster_algo=DBSCAN(eps=eval, min_samples=1)
                        cluster_data=cluster_algo.fit_predict(np_source)
                        cluster_centers=[[] for i in range(max(cluster_data)+1)]
                        for objnr, objcluster in enumerate(cluster_data):
                            cluster_centers[objcluster].append(np_source[objnr,0])
                        for clusternr in range(len(cluster_centers)):
                            cluster_centers[clusternr]=np.mean(cluster_centers[clusternr])
                        #print("cluster labels :", cluster_data)
                        #print("cluster centers :", cluster_centers)
                        cluster_to_plot=np.argmax(cluster_centers)
                        interestobjs=[idx for idx in range(len(cluster_data)) if cluster_data[idx]==cluster_to_plot]
                        if len(interestobjs)>clusterlenthr:
                            interestobjs=[]
                        if bboxcaplength!=None:
                            if len(interestobjs)>bboxcaplength:
                                interestobjs=np.flip(np.argsort(np_source, 0))[:,0][:bboxcaplength] #Get the bboxcaplength with highest values
                         
                         
                            
                elif pooling=="sum":
                    #lambdathr plot mode
                    if plotmode=="lambda_thr":
                        for obj in range(len(source[0])):
                            if np.sum(source,0)[obj]>val*maxvalue:
                                interestobjs.append(obj)
                                
                    #nbboxes plot mode
                    elif plotmode=="nbboxes":
                        np_source = np.sum(source[1:(nbwords+1),:],0)
                        sorted_objs= np.sort(np_source)
                        interestobjs = np.flip(np.argsort(np_source))[:val]
                        
                    #clustering plot mode
                    elif plotmode=="clustering":
                        np_source = np.sum(source[1:(nbwords+1),:],0)
                        np_source = np.expand_dims((np_source/max(np_source)), 1)
                        cluster_algo=DBSCAN(eps=val, min_samples=1)
                        cluster_data=cluster_algo.fit_predict(np_source)
                        cluster_centers=[[] for i in range(max(cluster_data)+1)]
                        for objnr, objcluster in enumerate(cluster_data):
                            cluster_centers[objcluster].append(np_source[objnr,0])
                        for clusternr in range(len(cluster_centers)):
                            cluster_centers[clusternr]=np.mean(cluster_centers[clusternr])
                        #print("cluster labels :", cluster_data)
                        #print("cluster centers :", cluster_centers)
                        cluster_to_plot=np.argmax(cluster_centers)
                        interestobjs=[idx for idx in range(len(cluster_data)) if cluster_data[idx]==cluster_to_plot]
                        if len(interestobjs)>clusterlenthr:
                            interestobjs=[]
                        if bboxcaplength!=None:
                            if len(interestobjs)>bboxcaplength:
                                interestobjs=np.flip(np.argsort(np_source, 0))[:,0][:bboxcaplength] #Get the bboxcaplength with highest values
                    

            #Deprecated for now
            if mode=='lr':
                for query in range(len(source)):
                    for obj in range(len(source[0])):
                        if source[query][obj]>val*maxvalue:
                            interestobjs.append((query,obj))
            

            # Create a Rectangle patch
            for i in interestobjs:
                if mode=='x':
                    bbox=boxes_unscaled[i]
                if mode=='lr':
                    bbox=boxes_unscaled[i[1]]
                ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none'))
                #ax1.add_patch(patches.Rectangle((bbox[2], bbox[3]), -55, 40, linewidth=1, edgecolor='r', facecolor='r'))
                #ax1.text(bbox[2]-55,bbox[3]+37, str(i))# + ", " + obj_cats[A['objects'][i]]['name'])
                
                if mode=='lr':
                    ax1.text(bbox[0],bbox[3], str(i))#+", q: "+str(queries[i[0]])+", cat: "+obj_cats[A['objects'][i[1]]]['name'])

            ax1.imshow(imgplot)
            ax1.set_title(img_to_plot)
            fig.canvas.draw_idle()

        # register the update function with each slider
        lmbd_slider.on_changed(update)
        ax1.set_title(img_to_plot)
        plt.show()




########################### VISN ATTSC
    if visn==1:
        head_to_plot=4
        source=sum([abs(np.transpose(attsc_layer_5_visn_currimg[i])) for i in range(nbheads)])
        #source=abs(attsc_layer_5_lang_currimg[head_to_plot-1])
        #print(np.shape(source))
        
        if mode=='x':
            if pooling=="CLS":
                maxvalue=np.amax(source[:, 0])
            elif pooling=="sum":
                maxvalue=np.amax(np.sum(source[:, 1:(nbwords+1)], 1))
        if mode=='lr':
            source=source[1:-1]
            maxvalue=np.amax(source)

        fig, (ax1, ax2) = plt.subplots(2,1)
        #img = ax.imshow(sum([abs(attsc_layer_2_lang_currimg[i]) for i in range(nbheads)]), extent=[0, 100, 0, 54], cmap='jet')
        #img = ax.imshow(abs(sum_layer_3_lang), extent=[0, 100, 0, 54], cmap='jet')
        img = ax2.imshow(source, extent=[0, 100, 0, 45], cmap='jet')
        
        y_label_list = [15,10,5,1]
        ax2.set_yticks([1.5,16.5,31.5,43.5]) #Positions on the y axis for ticks in y_label_list. Refer to img to know height bounds
        ax2.set_yticklabels(y_label_list)
        bar = fig.colorbar(img)
        strqueries=""
        for i in queries[(1*mode=="lr"):nbwords-(1*mode=='lr')]: #Get rid of [CLS] and [SEP] labels on axis for 'lr' mode
            strqueries+=i
            strqueries+="\n"
        ax2.set_ylabel(strqueries, rotation='horizontal')
        ax2.yaxis.set_label_coords(-0.12,-0.06)

        os.chdir("C:/Users/wantiez/Documents/FP")
        
        imgplot=mpimg.imread("./testclips_downsampled_6fpv/"+"_".join(img_to_plot.split("_")[:3])+"/"+img_to_plot+".jpg")
        img_w=imgplot.shape[1] #xmax
        img_h=imgplot.shape[0] #ymax
        
        A=np.load("./bdd100k/feature_output/test/testclips_downsampled_6fpv/"+"_".join(img_to_plot.split("_")[:3])+".npy", allow_pickle=True)[frame_nr] #Image to load
        
        boxes_unscaled=A['bbox']
        #Flip all bounding boxes
        for bbox_nr, bbox in enumerate(boxes_unscaled):
            #boxes_unscaled[bbox_nr]=flip_bbox_around_center(bbox, img_w, img_h)
            boxes_unscaled[bbox_nr]=bbox
        
        interestobjs=[]

        if mode=='x':
            if pooling=="CLS":
                #lambdathr plot mode
                if plotmode=="lambda_thr":
                    for obj in range(len(source[0])):
                        if source[0][obj]>lambda_thr*maxvalue:
                            interestobjs.append(obj)
                
                #nbboxes plot mode
                elif plotmode=="nbboxes":
                    np_source = np.array(source[0])
                    sorted_objs= np.sort(np_source)
                    interestobjs = np.flip(np.argsort(np_source))[:nbboxes]
                            
            elif pooling=="sum":
                #lambdathr plot mode
                if plotmode=="lambda_thr":
                    for obj in range(len(source[0])):
                        if np.sum(source[1:(nbwords+1),:],0)[obj]>lambda_thr*maxvalue:
                            interestobjs.append(obj)
                
                #nbboxes plot mode
                elif plotmode=="nbboxes":
                    np_source = np.sum(source[1:(nbwords+1),:],0)
                    sorted_objs= np.sort(np_source)
                    interestobjs = np.flip(np.argsort(np_source))[:nbboxes]

        ##Deprecated for now
        if mode=='lr':
            for query in range(len(source)):
                for obj in range(len(source[0])):
                    if source[query][obj]>lambda_thr*maxvalue:
                        interestobjs.append((query,obj))
        
        obj_cats=list(json.load(open("visual_genome_categories.json")).values())[0]
        
        im2=ax1.imshow(imgplot)
        # Create a Rectangle patch
        for i in interestobjs:
            if mode=='x':
                bbox=boxes_unscaled[i]
            if mode=='lr':
                bbox=boxes_unscaled[i[1]]
            ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none'))
            if mode=='x':
                ax1.text(bbox[0],bbox[3], str(i) + ", " + obj_cats[A['objects'][i]]['name'])

            if mode=='lr':
                ax1.text(bbox[0],bbox[3], str(i)+", q: "+str(queries[i[0]])+", cat: "+obj_cats[A['objects'][i[1]]]['name'])

        #
        axlbd = plt.axes([0.1, 0.15, 0.0225, 0.8])
        if plotmode=="lambda_thr":
            lmbd_slider = Slider(
                ax=axlbd,
                label='lambda_thr',
                valmin=0,
                valmax=1,
                valinit=lambda_thr,
                orientation='vertical'
            )

        elif plotmode=="nbboxes":
            lmbd_slider = Slider(
                ax=axlbd,
                label='nb_boxes',
                valmin=0,
                valmax=20,
                valinit=nbboxes,
                valstep=1,
                orientation='vertical'
            )

        # The function to be called anytime a slider's value changes
        def update(val):
            ax1.clear()
            interestobjs=[]

            if mode=='x':
                if pooling=="CLS":
                    #lambdathr plot mode
                    if plotmode=="lambda_thr":
                        for obj in range(len(source[0])):
                            if source[0][obj]>val*maxvalue:
                                interestobjs.append(obj)
                            
                    #nbboxes plot mode
                    elif plotmode=="nbboxes":
                        np_source = np.array(source[0])
                        sorted_objs= np.sort(np_source)
                        interestobjs = np.flip(np.argsort(np_source))[:val]
                    
                            
                elif pooling=="sum":
                    #lambdathr plot mode
                    if plotmode=="lambda_thr":
                        for obj in range(len(source[0])):
                            if np.sum(source,0)[obj]>val*maxvalue:
                                interestobjs.append(obj)
                                
                    #nbboxes plot mode
                    elif plotmode=="nbboxes":
                        np_source = np.sum(source[1:(nbwords+1),:],0)
                        sorted_objs= np.sort(np_source)
                        interestobjs = np.flip(np.argsort(np_source))[:val]

            #Deprecated for now
            if mode=='lr':
                for query in range(len(source)):
                    for obj in range(len(source[0])):
                        if source[query][obj]>val*maxvalue:
                            interestobjs.append((query,obj))
            

            # Create a Rectangle patch
            for i in interestobjs:
                if mode=='x':
                    bbox=boxes_unscaled[i]
                if mode=='lr':
                    bbox=boxes_unscaled[i[1]]
                ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none'))
                if mode=='x':
                    ax1.text(bbox[0],bbox[3], str(i) + ", " + obj_cats[A['objects'][i]]['name'])
                
                if mode=='lr':
                    ax1.text(bbox[0],bbox[3], str(i)+", q: "+str(queries[i[0]])+", cat: "+obj_cats[A['objects'][i[1]]]['name'])

            ax1.imshow(imgplot)
            ax1.set_title(img_to_plot)
            fig.canvas.draw_idle()

        # register the update function with each slider
        lmbd_slider.on_changed(update)
        ax1.set_title(img_to_plot)
        plt.show()

if ctx==1:
    if lang==1:
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(source.T, extent=[0,54, 0, 64], cmap='jet')
        x_label_list = [0,5,10,15]
        ax.set_xticks([0,15,30,45])
        y_label_list = [60, 50, 40, 30, 20, 10, 0]
        ax.set_yticks([4, 14, 24, 34, 44, 54, 64])
        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(y_label_list)
        bar = plt.colorbar(img)
        plt.title=('ctx_lang')
        plt.show()
    
    if visn==1:
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(sum([abs(ctxlay_layer_2_visn_currimg[i].T) for i in range(nbheads)]), extent=[0,100, 0, 64], cmap='jet')
        y_label_list = [0,15,30,45,60]  #NOT RIGHT BE CAREFUL (KNOW WHERE THE ORIGIN STARTS)
        ax.set_yticks([0,15,30,45,60])
        ax.set_yticklabels(y_label_list)
        bar = plt.colorbar(img)
        plt.title=('ctx_visn')
        plt.show()
    

    #Axis are not right -> differentiate plots for lang and for visn
    #Make sure to include heatmaps from all the different layers (1,2,3) of the model

