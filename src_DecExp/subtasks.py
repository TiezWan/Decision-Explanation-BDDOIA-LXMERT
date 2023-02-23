import os
import collections
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import datetime
import numpy as np
import pdb #For debugging
import random


from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from subtasks_model import SubtaskModel, LOGIT_ANSWER_LENGTH, QUERY_LENGTH
from subtasks_data import SubtaskDataset#, QUESTION_NR
from lxrt.optimization import BertAdam



import warnings
warnings.filterwarnings("ignore") #Attempts to ignore deprecated warnings from Pytorch, as those haven't been fixed in this torch version

class Subtask:
    """Imports the dataset(s) and builds the dataloader and model. Performs training and evaluation with parameters defined in an
    external args variable."""
    def __init__(self):
        #torch.cuda.set_device(4)

        # Train and validation datasets
        if args.test!=True:
            self.trainset = SubtaskDataset(args.train)
            print("Number of samples in training set :", self.trainset.__len__())

            if args.valid!=None and args.valid !="":
                self.validset=SubtaskDataset(args.valid)
                print("Number of samples in validation set :", self.validset.__len__())


        # Output Directory
        self.output = args.output #For later reference
        os.makedirs(args.output, exist_ok=True)

        # Model
        self.model = SubtaskModel()
        print("Model built \n")

        # Loading pre-trained weights
        if args.load_lxmert is not None: #pre-trained weights
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None: #fine-tuned wieghts with QA answer head
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                        label2ans=self.train_tuple.dataset.label2ans)
        
        # Loading fine-tuned weights from previous training sessions
        if args.load is not None: 
            self.load(args.load)
        ###Selectively freeze layers:
        # for rlayer in range(args.rlayers):
        #     self.model.lxrt_encoder.model.bert.encoder.r_layers[rlayer].attention.self.tempquery.weight.requires_grad=False
        #     self.model.lxrt_encoder.model.bert.encoder.r_layers[rlayer].attention.self.tempkey.weight.requires_grad=False
        #     self.model.lxrt_encoder.model.bert.encoder.r_layers[rlayer].attention.self.tempvalue.weight.requires_grad=False


        ###Otherwise, freeze everything and unfreeze specific layers
        # for param in self.model.parameters():
        #     param.requires_grad=False
        # ###And selectively unfreeze layers after this
        # self.model.logit_fc[0].weight.requires_grad=True #0 corresponds to the first Linear layer
        # self.model.logit_fc[3].weight.requires_grad=True #3 corresponds to the second linear layer

        #Testing process
        if args.test==True:
            args.img_num=1e12 #Always test on full test set
            args.dotrain=False
            self.testset=SubtaskDataset(args.test)
            print("Number of samples in test set :", self.testset.__len__())

            #Perform evaluation only
            # test_accuracy, avg_test_loss=decexp.evaluate(decexp.testset)
            # print("Test accuracy", test_accuracy*100., "Avg loss: ", avg_test_loss)
            # with open(args.output + '/test_acc_model_n_n_n', 'a') as f:
            #         f.write("\n\nModel: ")
            #         f.write(args.load)
            #         f.write("\nTolerance: 0.5")
            #         f.write("\nNumber of needed bits: 25")
            #         f.write("\nHeads: 12")
            #         f.write("\nTest accuracy: ")
            #         f.write(str(test_accuracy*100.))
            #         f.write(" Avg loss: ")
            #         f.write(str(avg_test_loss))
            #         f.flush()


        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        print("batch_size:", args.batch_size)
        #self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=clweight, reduction='none') #Loss for multi-label multi-class classification. Choice: mean of the outputs
        #clweight=torch.ones([LOGIT_ANSWER_LENGTH])
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.batches_per_epoch=int(np.ceil(self.trainset.__len__()/args.batch_size))
        print("Batches per epoch: ", self.batches_per_epoch)
        #t_total = int(self.batches_per_epoch * args.epochs)
        t_total=args.epochs*self.trainset.__len__()
        print("Total number of batches: %d" % t_total)

        if 'bert' in args.optim:
            self.optim = BertAdam(list(self.model.parameters()),
                                lr=args.lr,
                                warmup=0.1,
                                t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        self.queries=json.load(open("../input/bdd100k/questionSet.json", 'r'))
        self.question_types=list(self.queries.keys())
        
    def train(self):
        print("Starting training")

        obj_cats=list(json.load(open("visual_genome_categories.json")).values())[0]

        best_valid = 0.
        best_bitwise_valid=0.
        best_F1=0.
        self.valacc_hist={}
        self.loss_hist={}
        self.val_loss_hist={}
        self.bitwise_valacc_hist={}
        self.valF1_hist={}

        for epoch in range(1,args.epochs+1):
            print("epoch ", epoch, "/", args.epochs, "\n")
            idx2ans = {}
            temp_loss_hist=[]
            if args.save_heatmap==True:
                train_loader = DataLoader(
                    self.trainset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    drop_last=True, pin_memory=False)
            else:
                train_loader = DataLoader(
                    self.trainset, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers,
                    drop_last=True, pin_memory=False) #Set pin_memory to True for faster training -> Leads to obnoxious unsolvable warnings
            #train_loader is a list of lists/tensors of batch_size elements (see what __getitem__ returns in DecExp_data): 
            #[[2,56,84,10,...], [aded54c2-98d4e31_2, jde9d9e7d-dgd232df_3, ...], [100, 100, ...],
            #tensor([[[0.0000, 0.0000, 2.1567, ...], [0.0000, 0.0000, 0.0000, ...], ... ]]) of shape (1 x obj_num x 1024)
            #tensor([[[0.5985, 0.1576, 0.6792, 0.3128], [0.3612, 0.4612, 0.6210, 0.8215], ...]]) of shape (1 x obj_num x 4)
            #tensor([[0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 1., 0., 0., 0., 1., 0.],...]])]
            
            trainbatch=0
            barloader=tqdm(train_loader, total=self.batches_per_epoch)
            for batchdata in barloader:
                trainbatch+=1
                #print("Training batch ",trainbatch, "/", self.batches_per_epoch)
                idx, pre_img_id, obj_num, feats, boxes, objs, frame_nr, prelabels = batchdata
                
                img_id=[]
                for i in range(len(pre_img_id)):
                    img_id.append(pre_img_id[i]+"_frame_"+str(frame_nr[i].item()))
                
                #idx=idx.tolist()
                #____________________________________
                #For one batch
                self.model.train() #Tells the model that we are in train mode
                self.optim.zero_grad() #Reset all gradients to 0
                feats, boxes, objs = feats.cuda(), boxes.cuda(), objs.cuda() #label is the logits after sigmoid
                
                self.sent=[]
                #type of question
                question_nr=[]
                #question or counter-question
                ques_appendix=[]
                
                for batch_item in range(args.batch_size):
                    #available_questions=[index for index,value in enumerate(prelabels[batch_item]) if ((value==0. or value==1.) and index%2==0)]
                    available_questions=[0,4,6]#0 for redlights, 4 for greenlights, 6 for road signs
                    question_nr.append(random.choice(available_questions))
                    ques_appendix.append(random.choice([0,1]))
                    self.sent.append(random.choice(self.queries[self.question_types[question_nr[-1]+ques_appendix[-1]]]))
                
                logits = self.model(img_id, feats, boxes, self.sent) #Does the model return logits or labels ? -> logits
                
                labels=torch.tensor([prelabels[i,question_nr[i]+ques_appendix[i]] for i in range(args.batch_size)], dtype=torch.float32).unsqueeze(1).cuda()
                
                #for backwardspass in range(text_logits.size(0)):
                    #l2_norm = sum([p.pow(2.0).sum() for p in self.model.parameters() if p.requires_grad])
                loss_unreduced = self.bce_loss(logits, labels)
                    #print(f"loss={loss_unreduced}")
                    
                loss= torch.mean(loss_unreduced, 0) #+ args.l2reg * l2_norm
                    
                    #print("loss :", loss.item(), "\n")
                    #loss = loss * logit.size(1) #Removing the mean reduction of bce_loss to get the sum of all components
                                                #Uncomment if the loss is chosen without specifying the reduction (defaults to mean)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                self.optim.zero_grad()
                
                str_loss="loss: {0:.3f}".format(loss[-1].item())
                barloader.set_description(str_loss)
                temp_loss_hist.append(loss[-1].item())
                
                #for indx,l in zip(idx, labels.cpu().numpy()):
                #    idx2ans[indx]=l #Make sure that imgid is in quotes to be used in DecExpEvaluate. idx2ans = {idx1:label1, ..., idxn:labeln}
                #temp_loss_hist.append(loss.item())
                #self.loss_hist[epoch]=loss.item() #Save the loss at the end of each epoch for later plotting along with the epoch number as key
                #str_loss="loss: 10e-3 * {0:.3f}".format(loss.item()*1000)
                #barloader.set_description(str_loss)
                
            
            self.loss_hist[epoch]=temp_loss_hist
            #print(self.loss_hist[epoch])

            with open(args.output + "/losshist.log", 'a') as f:
                str_loss=str(self.loss_hist[epoch])
                f.write("loss history epoch ")
                f.write(str(epoch))
                f.write(": ")
                f.write(str_loss)
                f.write("\n")
                f.flush()
            self.save('epoch_%d'% (epoch)) #Saves the weights for each epoch for later (re-)evaluation

            
            train_accuracy, bitwise_train_accuracy, trainloss, train_trace, trainF1, train_conf_mat=self.evaluate(self.trainset, self.sent) #Computes training accuracy
            log_str="\nTraining --- \n  Label-wise accuracy: Epoch {0}: {1} % \n  Bit-wise accuracy: Epoch {0}: {2} \n  Bit-wise F1-score: Epoch {0}: {3} \n  Confusion matrix: Epoch {0}: {4}\n".format(epoch, train_accuracy*100., bitwise_train_accuracy*100., trainF1, train_conf_mat)

            with open(args.output + "/trainhist.log", 'a') as f:
                f.write("\nEpoch: ")
                f.write(str(epoch))
                f.write(f"\n training accuracy: {train_accuracy}")
                f.write(f"\n training loss: {trainloss}")
                f.write(f"\n training F1 score: {trainF1}")
                f.write(f"\n training confusion matrix: {train_conf_mat}\n")
                
            with open(args.output + "/traintrace.log", 'a') as f:
                f.write(str(epoch))
                f.write(": ")
                f.write(str(train_trace))
                f.write("\n")


            if args.valid != "" and args.valid != None:
                val_accuracy, bitwise_val_accuracy, val_loss, val_trace, valF1, val_conf_mat=self.evaluate(self.validset, self.sent) #Compute validation accuracy
                self.valacc_hist[epoch]=val_accuracy #Save the validation accuracy from the current epoch for later plotting
                self.val_loss_hist[epoch]=val_loss
                self.bitwise_valacc_hist[epoch]=bitwise_val_accuracy
                self.valF1_hist[epoch]=valF1
                if val_accuracy > best_valid: #If the validation accuracy from this epoch is better than the current best from a previous epoch, save it
                    best_valid = val_accuracy
                    self.save("BEST_val")
                if bitwise_val_accuracy > best_bitwise_valid:
                    best_bitwise_valid=bitwise_val_accuracy
                if valF1 > best_F1:
                    best_F1=valF1

                log_str += "Validation --- \n  Label-wise accuracy: \n    Epoch {0}: {1} %\n    Epoch {0}: Best {2} %\n  Bit-wise accuracy: \n\
    Epoch {0}: {3} %\n    Epoch {0} Best Bit-wise accuracy {4} %\n  Bit-wise F1-score: \n    Epoch {0}: {5} \n    Epoch {0}\
 Best Bit-wise F1-score {6} \n  Confusion matrix: {7} \n".format(epoch, val_accuracy*100., best_valid*100., bitwise_val_accuracy*100., best_bitwise_valid*100., valF1, best_F1, val_conf_mat)
            
                with open(args.output + "/valhist.log", 'a') as f:
                    f.write(log_str)
                    f.flush()

                with open(args.output + "/val_loss_hist.log", 'a') as f:
                    str_val_loss=str(val_loss)
                    f.write("Average validation loss epoch ")
                    f.write(str(epoch))
                    f.write(": ")
                    f.write(str_val_loss)
                    f.write("\n")
                    f.flush()

                print("val_loss: ",val_loss)

                with open(args.output + "/valtrace.log", 'a') as f:
                    f.write(str(epoch))
                    f.write(": ")
                    f.write(str(val_trace))
                    f.write("\n")

            print(log_str)
        
        #date=str(datetime.datetime.now())[:19].replace('-', '_').replace(' ', '-').replace(':', '_')
        #self.save('DecExp_train_weights_%s' %date)


    def predict(self, eval_loader, eval_batches, sents, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        idx2anseval = {}
        idxtoquestype={}
        batchval=0
        
        for data in tqdm(eval_loader, total=eval_batches):
            batchval+=1
            idx, pre_img_id, obj_num, feats, boxes, objs, frame_nr = data[0:7]
            img_id=[]
            for i in range(len(pre_img_id)):
                img_id.append(pre_img_id[i]+"_frame_"+str(frame_nr[i].item()))
            
            obj_cats=list(json.load(open("visual_genome_categories.json")).values())[0]
            
            sents=[]
            #type of question
            question_nr=[]
            #question or counter-question
            ques_appendix=[]
            for batch_item in range(args.batch_size):
                available_questions=[0,4,6] #0 for redlights, 4 for greenlights, 6 for road signs
                if args.batch_size!=1:
                    question_nr.append(random.choice(available_questions))
                    ques_appendix.append(random.choice([0,1]))
                    sents.append(random.choice(self.queries[self.question_types[question_nr[-1]+ques_appendix[-1]]]))
                
                elif args.batch_size==1:
                    for i in available_questions:
                        sents.append(random.choice(self.queries[self.question_types[i]])) #Question i
                        #sents.append(random.choice(self.queries[self.question_types[i+1]])) #Counter-question i
            
            if args.batch_size==1:
                feats=feats.repeat(len(sents), 1, 1)
                boxes=boxes.repeat(len(sents), 1, 1)
                img_id=[img_id[0] for i in range(len(sents))]
            
            with open(args.output + "/sentstrace.log", 'a') as f:
                f.write(str(sents))
                f.write("\n")
                f.write(str(img_id))
                f.write("\n")
            
            idx=idx.tolist()
            torch.cuda.empty_cache()
            with torch.no_grad(): #Making sure we don't do any training here
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(img_id, feats, boxes, sents)
                
                label=nn.Sigmoid()(logit)
                if args.batch_size!=1:
                    for index, l in zip(idx, label.cpu().numpy()):
                        idx2anseval[index] = np.array(l)
                else:
                    idx2anseval[idx[0]]=label
                
                if args.batch_size!=1:
                    for index, qtype in zip(idx, [question_nr[i]+ques_appendix[i] for i in range(len(question_nr))]):
                        idxtoquestype[index]=[qtype]
                elif args.batch_size==1:
                    for index in idx:
                        idxtoquestype[index]=available_questions
                
            #if dump is not None:
            #    self.dump_result(imgid2ans, dump)
        if args.dotrain==False: #In eval mode
            args.output+="/eval_tasks_"
            args.output+="_".join([str(i) for i in available_questions])
        return img_id, idx2anseval, idxtoquestype, sents

    def evaluate(self, dset, sents, dump=None):
        """Evaluate all data in data_tuple."""
        print("\n Starting evaluation")
        sents=[]
        self.val_loss={}
        #if args.img_num!=None:
        #    val_batches=int(np.ceil(args.img_num/args.batch_size))
        #else:
        eval_batches=int(np.ceil(dset.__len__()/args.batch_size))
        print("Number of evaluation batches:", eval_batches)
        eval_answers={}
        score = 0.
        tolerance=0.5 #We accept a confidence level of 1-0.5=0.5, that is 50% (Binary classification)
        nbequal=LOGIT_ANSWER_LENGTH #number of bits within each prediction that have to be equal to the label in order to classify as correct.
        
        if args.save_heatmap==True:
            eval_loader = DataLoader(
                dset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                drop_last=True, pin_memory=False)
        
        else:
            eval_loader = DataLoader(
                dset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                drop_last=True, pin_memory=False)
            
        print("Dataloader loaded")
        imgids, temp, idxtoquestype, sents=self.predict(eval_loader, eval_batches, sents, dump)
        eval_answers.update(temp)
        
        if args.save_predictions==True:
            with open(args.output + "/predictions.log", 'a') as f:
                for idx in eval_answers.keys():
                    f.write(", ".join(sents))
                    f.write(",  ")
                    f.write(str(dset.idx2label[idx][0]))
                    f.write(", ")
                    f.write(str(dset.idx2label[idx][1]))
                    f.write(", ")
                    f.write(str(eval_answers[idx]))
                    f.write("\n")
        
        #get labels for the idx chosen by the dataloader from the dataset annotations
        
        labels={}
        for indx in temp.keys():
            try:
                if args.batch_size!=1:
                    labels[indx]=(np.expand_dims(np.array(dset.idx2label[indx][1][idxtoquestype[indx]]), 0))
                elif args.batch_size==1:
                    labels[indx]=np.array([dset.idx2label[indx][1][i] for i in idxtoquestype[indx]])
            except:
                pdb.set_trace()
        
        
        del temp
        trace=np.zeros(len(eval_answers[list(eval_answers.keys())[0]])) #Count position-wise errors over the dataset
        fullques_score=0
        bitwise_score=0
        Falsepos=0
        Truepos=0
        Falseneg=0
        Trueneg=0
        
        for indx in eval_answers.keys():
            temp=0
            tempactions=0
            for bit in range(len(eval_answers[indx])):
                if eval_answers[indx][bit]>=labels[indx][bit]-tolerance and eval_answers[indx][bit]<labels[indx][bit]+tolerance:
                    bitwise_score+=1
                    temp+=1
                else:
                    trace[bit]+=1 #Keeps track of bits that are incorrectly predicted
                
                Falsepos+=int(eval_answers[indx][bit]+0.5)*(1-int(labels[indx][bit]))
                Truepos+=int(eval_answers[indx][bit]+0.5)*int(labels[indx][bit])
                Falseneg+=(1-int(eval_answers[indx][bit]+0.5))*int(labels[indx][bit])
                Trueneg+=(1-int(eval_answers[indx][bit]+0.5))*(1-int(labels[indx][bit]))

            #if np.allclose(eval_answers[indx],labels[indx], rtol=0, atol=tolerance):
                #score+=1
            target_shape=(torch.tensor(eval_answers[indx], dtype=torch.float32).shape[0],1) #labels is of shape [n,1] while answers is of shape [n]

            self.val_loss[indx] = nn.BCELoss()(torch.tensor(eval_answers[indx], dtype=torch.float32).reshape(target_shape).cuda(), 
                torch.tensor(labels[indx], dtype=torch.float32).reshape(target_shape).cuda()).item()
            # elif args.batch_size==1:
            #     self.val_loss[indx] = nn.BCELoss()(torch.tensor(eval_answers[indx][:,0], dtype=torch.float32), 
            #         torch.tensor(labels[indx], dtype=torch.float32).cuda()).item() #Batch_size==1 requires a squeeze in the answers array
            
            #if isinstance(idxtoquestype[indx], int):
            #    fullques_score+=temp
            if temp>=len(idxtoquestype[indx]): #If a minimum number of bits correspond between prediction and annotation, score +=1
                                #NOT THE SAME AS ADDING 1 TO SCORE IF A BIT CORRESPONDS (i.e computing the % of correct bits overall)
                fullques_score+=1
        
        conf_mat=(Trueneg, Falsepos, Falseneg, Truepos)
        #print(trace)
        listlosses=list(self.val_loss.values())
        avg_eval_loss=sum(listlosses)/len(listlosses)
        #actionlabel_acc=actionlabel_score/(len(eval_answers)*4)
        label_acc=fullques_score/len(eval_answers)
        bitwise_acc=bitwise_score/(len(eval_answers)*len(eval_answers[list(eval_answers.keys())[0]]))
        try:
            F1score=Truepos/(Truepos+0.5*(Falsepos+Falseneg))
        except:
            print(Falsepos, Truepos, Falseneg, Trueneg)
            F1score=1.
            pdb.set_trace()
        try:
            print("\nprecision: ", (Truepos/(Truepos+Falsepos)))
            print("\nrecall: ", (Truepos/(Truepos+Falseneg)))
            #print("Bitwise action accuracy: ", actionlabel_acc*100.)
        except:
            print("error while computing precision, recall or accuracy")
            pdb.set_trace()
        return label_acc, bitwise_acc, avg_eval_loss, trace, F1score, conf_mat



    def save(self, name):
        torch.save(self.model.state_dict(),
                os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
        print("Weights loaded")


if __name__ == "__main__":
    # Set process start-method
    # from torch.multiprocessing import set_start_method
    # try:
    #     set_start_method('spawn')
    # except:
    #     print("cannot set start method to 'spawn'")
    #     sys.exit()
    # Build Class
    args.img_num=None
    args.train='train'
    args.valid='val'
    args.test=None
    args.batch_size=1 #Batch_size of 1 reserved for sending all queries to 1 image at a time
    args.epochs=40
    #args.output='./snap/no_clweight_reg_0p0003_train_5000_12h_3_3_3_bs8_feats100'
    args.output='./subtasks/snap_subtasks/bdd100k/12_9_5_5_bs8_mixed_tasks_5em5_corrected_pooler_shuffle'
    args.lr=5e-5
     
    #Load subtask fine-tuned model weights. Note: It is different from loading LXMERT pre-trained weights.
    args.load='./subtasks/snap_subtasks/bdd100k/12_9_5_5_bs8_mixed_tasks_5em5_corrected_pooler_shuffle/epoch_19' 
    
    #args.load_lxmert='./snap/model' #load LXMERT pre-trained weights
    #args.from_scratch=True
    args.save_predictions=False
    args.save_heatmap=False
    args.num_workers=0
    
    args.dotrain=False #True: trains the model. False: Performs evaluation only
    args.heads=12
    args.llayers=9
    args.xlayers=5
    args.rlayers=5
    args.l2reg=0.

    subtask = Subtask()

    if args.test!=True:
        if args.dotrain==False:
            #Perform evaluation only
            print("Performing evaluation only")
            #train_accuracy, avg_train_loss=subtask.evaluate(subtask.trainset)
            for i in range(1,2):
                #with torch.no_grad():
                    #print("loading epoch", i)
                    #subtask.load('./snap/train_full_3h_3_3_3_x/epoch_'+str(i))
                eval_label_accuracy, eval_bitwise_accuracy, avg_eval_loss, eval_trace, evalF1, conf_mat=subtask.evaluate(subtask.validset, [])
                #print("Train accuracy", train_accuracy*100., "Avg loss: ", avg_train_loss)
                print("Val accuracy", eval_label_accuracy*100., "Avg loss: ", avg_eval_loss)

                log_str = "Evaluation --- \n  Label-wise accuracy: \n    Epoch {0}: {1} %\n  Bit-wise accuracy: \n    Epoch {0}: {2} %\n\
  Bit-wise F1-score: \n    Epoch {0}: {3} \n".format(i, eval_label_accuracy*100., eval_bitwise_accuracy*100., evalF1)
                print(log_str)

                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                with open(args.output + "/eval_acc_loss.log", 'a') as f:
                        f.write("Model: ")
                        if args.load!=None:
                            f.write(args.load)
                        elif args.load_lxmert!=None:
                            f.write(args.load_lxmert)
                        #f.write("\nTrain accuracy: ")
                        #f.write(str(train_accuracy*100.))
                        #f.write(" Avg loss: ")
                        #f.write(str(avg_train_loss))
                        f.write("\n")
                        f.write(f"{args.valid} set")
                        f.write("\nEval label accuracy: ")
                        f.write(str(eval_label_accuracy*100.))
                        f.write("\nEval bitwise accuracy: ")
                        f.write(str(eval_bitwise_accuracy*100.))
                        f.write("\nEval bitwise F1-score: ")
                        f.write(str(evalF1))
                        f.write("\nConfusion matrix")
                        f.write(str(conf_mat))
                        f.write("\nAvg loss: ")
                        f.write(str(avg_eval_loss))
                        f.write("\n \n")
                        f.flush()

                with open(args.output + "/evaltrace.log", 'a') as f:
                    f.write(str(i))
                    f.write(": ")
                    f.write(str(eval_trace))
                    f.write("\n")

        else: #Call the training function
            print("Splits in data: \n", subtask.trainset.split)
            if args.valid!=None and args.valid!="":
                print("", subtask.validset.split, "\n")
            subtask.train()
        
        

