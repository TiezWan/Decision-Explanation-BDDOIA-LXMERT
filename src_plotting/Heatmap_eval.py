import numpy as np
import json
import pdb
import os
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['axes.facecolor']=[0.9, 0.9, 0.9]
# matplotlib.rcParams['axes.labelsize']=14
# matplotlib.rcParams['axes.titlesize']=14
# matplotlib.rcParams['xtick.labelsize']=12
# matplotlib.rcParams['ytick.labelsize']=12
# matplotlib.rcParams['legend.fontsize']=12
# matplotlib.rcParams['legend.facecolor']='w'
# matplotlib.rcParams['savefig.transparent']=False
#config InlineBackend.figure_format = 'retina'
import seaborn as sns
sns.set_theme()

with open("bdd100k_heatmaps_testlabels_redlights.json", 'r') as f:
    labels_redlights=json.load(f)
    
with open("bdd100k_heatmaps_testlabels_greenlights.json", 'r') as f:
    labels_greenlights=json.load(f)
    
with open("bdd100k_heatmaps_testlabels_roadsigns.json", 'r') as f:
    labels_roadsigns=json.load(f)
    
# dellistkeys=[]
# dellistkeysframes=[]
# for key in labels_redlights.keys():
#     #if key=='2012cb38-0163c056_0_40': pdb.set_trace()
#     for frame in labels_redlights[key].keys():
#         if labels_redlights[key][frame]==[None]:
#             dellistkeysframes.append((key, frame)) #Checking which frames contain None value

# for tuple in dellistkeysframes:
#     del labels_redlights[tuple[0]][tuple[1]] # Deleting them   

# for key in labels_redlights.keys():
#     if not bool(labels_redlights[key]): #Checking which keys now have empty values
#         dellistkeys.append(key) 
        
# for key in dellistkeys:
#     del labels_redlights[key]

labels_redlights_flat={}
for key in labels_redlights:
    for frame in labels_redlights[key]:
        labels_redlights_flat[frame]=labels_redlights[key][frame]
        
labels_greenlights_flat={}
for key in labels_greenlights:
    for frame in labels_greenlights[key]:
        labels_greenlights_flat[frame]=labels_greenlights[key][frame]
        
labels_roadsigns_flat={}
for key in labels_roadsigns:
    for frame in labels_roadsigns[key]:
        labels_roadsigns_flat[frame]=labels_roadsigns[key][frame]

labels_all=[labels_redlights_flat, labels_greenlights_flat, labels_roadsigns_flat]

nbheads=12
imgdir=f"12_9_5_5_bs8_mixed_tasks_5em5_corrected_pooler_shuffle_epoch_19/allsents_per_sample/test_set/seed_158467"
available_questions=[0,1,2]
nbobjs=100

mode="x"
pooling="sum"
plotmode="clustering"  #"nbboxes", "lambda_thr" or "clustering"
lambda_thr=0.8

##Clustering parameters
eps_dbscan=0.05
bboxcaplength=None #Maximum number of bbox to plot in clustering mode
clusterlenthr=5 #If the cluster of highest value objects has a length bigger than this, plot nothing
                #This choice is justified by the working of DBSCAN: if the cluster contains a lot of
                #objects, it is likely that there are lots of objects unrelated to the question

def compute_objs(imgid, eps_dbscan=0.05, plotmode="clustering", pooling="sum", available_questions=[0,1,2], nbboxes=5, lambda_thr=0.8, clusterlenthr=35):
    interestobjs=[] #Will contain objects of interest for all questions
    uninterestobjs=[]

    for question in available_questions:
        attsc_layer_5_lang_currimg=[]
        for i in range(nbheads):
            attsc_layer_5_lang_currimg.append(np.load("./bdd100k_heatmaps/"+imgdir+"/"+imgid \
                + "/sent"+str(question)+"/attention_scores/lang/layer_5/"+imgid+"_attsc_layer_5_lang_head_"+ str(i+1)+".npy"))

        source=sum([abs(attsc_layer_5_lang_currimg[i]) for i in range(nbheads)])

        with open(f"./bdd100k_heatmaps/{imgdir}/{imgid}/sent{question}/sents.txt", 'r') as f:
            sent=f.read()
            queries=sent.split(" ")
            nbwords=len(queries)
        
        if pooling=="CLS":
            maxvalue=np.amax(source[0])
            #lambdathr plot mode
            if plotmode=="lambda_thr":
                interestobjs_temp=[]
                for obj in range(len(source[0])):
                    if source[0][obj]>lambda_thr*maxvalue:
                        interestobjs_temp.append(obj)
            
            #nbboxes plot mode
            elif plotmode=="nbboxes":
                np_source = np.array(source[0])
                sorted_objs= np.sort(np_source)
                interestobjs_temp = np.flip(np.argsort(np_source))[:nbboxes]
                
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
                interestobjs_temp=[idx for idx in range(len(cluster_data)) if cluster_data[idx]==cluster_to_plot]
                if len(interestobjs_temp)>clusterlenthr:
                    interestobjs_temp=[]


        elif pooling=="sum":
            maxvalue=np.amax(np.sum(source[1:(nbwords+1),:], 0))
            #lambdathr plot mode
            if plotmode=="lambda_thr":
                interestobjs_temp=[]
                for obj in range(len(source[0])):
                    if np.sum(source[1:(nbwords+1),:],0)[obj]>lambda_thr*maxvalue:
                        interestobjs_temp.append(obj)
            
            #nbboxes plot mode
            elif plotmode=="nbboxes":
                np_source = np.sum(source[1:(nbwords+1),:],0)
                sorted_objs= np.sort(np_source)
                interestobjs_temp = np.flip(np.argsort(np_source))[:nbboxes]
            
            #clustering plot mode 
            elif plotmode=="clustering":
                np_source = np.sum(source[1:(nbwords+1),:],0) #The source is the column-wise sum of head-wise sum absolute values
                np_source = np.expand_dims((np_source/max(np_source)), 1)
                cluster_algo=DBSCAN(eps=eps_dbscan, min_samples=1) #Initializing the clustering algorithm
                cluster_data=cluster_algo.fit_predict(np_source) #Get the cluster for every point
                
                #Compute the value for the cluster centers, to determine which cluster to plot
                cluster_centers=[[] for i in range(max(cluster_data)+1)]
                for objnr, objcluster in enumerate(cluster_data):
                    cluster_centers[objcluster].append(np_source[objnr,0])
                for clusternr in range(len(cluster_centers)):
                    cluster_centers[clusternr]=np.mean(cluster_centers[clusternr])
                #print("cluster labels :", cluster_data)
                #print("cluster centers :", cluster_centers)
                cluster_to_plot=np.argmax(cluster_centers)
                
                #Get the objects in this cluster
                interestobjs_temp=[idx for idx in range(len(cluster_data)) if cluster_data[idx]==cluster_to_plot]
                
                if len(interestobjs_temp)>clusterlenthr:
                    interestobjs_temp=[]
        
        interestobjs.append(interestobjs_temp)
        
    return interestobjs
        
def evaluate(interestobjs, labels, available_questions, nbobjs, listimgids):
    ###
    #interestobjs is a numpy array of size (3, nb_img, nb_objs) 
    truepos=0
    falsepos=0
    falseneg=0
    trueneg=0
    precision=0.
    recall=0.
    acc=0.
    f1=0.
    #Computing overall accuracy based on all questions:
    for image_nr, image_id in enumerate(listimgids):
        for question in available_questions:
            for objnr in range(nbobjs):
                if (objnr in interestobjs[question][image_nr]) and (objnr in labels[question][image_id]):
                    truepos+=1
                elif (objnr in interestobjs[question][image_nr]) and not (objnr in labels[question][image_id]):
                    falsepos+=1
                elif not (objnr in interestobjs[question][image_nr]) and (objnr in labels[question][image_id]):
                    falseneg+=1
                elif not (objnr in interestobjs[question][image_nr]) and not (objnr in labels[question][image_id]):
                    trueneg+=1
                    
    conf_mat=np.array([trueneg, falsepos, falseneg, truepos])
          
    acc=(truepos+trueneg)/(truepos+trueneg+falsepos+falseneg)
    try:
        precision=(truepos/(truepos+falsepos))
        recall=(truepos/(truepos+falseneg))
        f1=truepos/(truepos+0.5*(falsepos+falseneg))
    except:
        pdb.set_trace()
        
    return acc, f1, precision, recall

def pad_list(list_to_pad, padlength, token):
    if type(list_to_pad)==np.ndarray:
        if len(list_to_pad.shape)==1:
            list_to_pad=list_to_pad.tolist()
            
    while len(list_to_pad)<padlength:
        list_to_pad.append(token)
    return list_to_pad
        
    
    
listimgids=os.listdir(f"./bdd100k_heatmaps/{imgdir}")
results={}
F1list=[]
acclist=[]
preclist=[]
reclist=[]

epslist=[i/200 for i in range(1, 41)]
nbboxes=[i for i in range(1,21)]
lambda_thrs=[i/40 for i in range(20, 41)]
lambda_thrs[-1]=0.999 #Cannot be equal to 1 or the calculation of f1 breaks down
for clusterlenthr in range(30,40, 10):
    results={}
    F1list=[]
    acclist=[]
    preclist=[]
    reclist=[]
    for param in epslist:
        np_interestobjs=np.zeros([len(available_questions), len(listimgids), nbobjs])

        for img_nr, img_id in enumerate(listimgids):
            interestobjs_img = compute_objs(img_id, eps_dbscan=param, plotmode="clustering", clusterlenthr=clusterlenthr)
            for question in range(len(interestobjs_img)):
                interestobjs_img[question]=pad_list(interestobjs_img[question], nbobjs, -1) #Padding with -1 up to a length of nbobjs
                np_interestobjs[question][img_nr]=interestobjs_img[question]
            
        #Get interestobjs for all questions and all images before calling evaluate
        acc, f1, precision, recall = evaluate(np_interestobjs, labels_all, available_questions, nbobjs, listimgids)

        results[param]=(acc, f1, precision, recall)
        F1list.append(f1)
        acclist.append(acc)
        preclist.append(precision)
        reclist.append(recall)
        
        print(results[param])

# with open("Eval_run_clustering_sum_eps_0p01_to_0p2.json", 'w') as f:
#     json.dump(results, f)
#     print("Successfully saved")

    plt.plot(epslist, acclist, label="Accuracy")
    plt.plot(epslist, F1list, label="F1 score")
    plt.plot(epslist, preclist, label="precision")
    plt.plot(epslist, reclist, label="recall")
    title=f"Eval_run_clustering_sum_clusterlenthr_{clusterlenthr}"
    plt.title(title)
    plt.legend()
    plt.show()