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
from DecExp_model import DecExpModel, ANSWER_LENGTH, QUERY_LENGTH
from DecExp_data import DecExpDataset
from lxrt.optimization import BertAdam

LOSSRATIODEC2EXP=5.

import warnings
warnings.filterwarnings("ignore") #Attempts to ignore deprecated warnings from Pytorch, as those haven't been fixed in this torch version

class DecExp:
    """Imports the dataset(s) and builds the dataloader and model. Performs training and evaluation with parameters defined in an
    external args variable."""
    def __init__(self):
        #torch.cuda.set_device(4)

        # Train and validation datasets
        if args.test!=True:
            self.trainset = DecExpDataset(args.train)
            print("Number of samples in training set :", self.trainset.__len__())
            
            #Computations of the correlation matrix for each set
            #imgidlabel=list(self.trainset.idx2label.values())
            #for i in range(len(imgidlabel)):
                #imgidlabel[i]=imgidlabel[i][1]
            
            #self.traincorr=np.corrcoef(imgidlabel)


            if args.valid!=None and args.valid !="":
                self.validset=DecExpDataset(args.valid)
                print("Number of samples in validation set :", self.validset.__len__())
                #imgidlabel=list(self.validset.idx2label.values())
                #for i in range(len(imgidlabel)):
                    #imgidlabel[i]=imgidlabel[i][1]
                #self.valcorr=np.corrcoef(imgidlabel)

            #Saving the correlation matrix to a txt and a npy file
            #np.savetxt("./snap/traincorr.txt", self.traincorr, fmt='%.3e')
            #np.savetxt("./snap/valcorr.txt", self.valcorr, fmt='%.3e')
            #np.save("./snap/traincorr.npy", self.traincorr)
            #np.savetxt("./snap/valcorr.npy", self.valcorr)

        # Output Directory
        self.output = args.output #For later reference
        os.makedirs(args.output, exist_ok=True)

        # Model
        self.model = DecExpModel()
        print("Model built \n")

        # Load pre-trained weights
        if args.load_lxmert is not None: #pre-trained weights
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None: #fine-tuned wieghts with QA answer head
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                        label2ans=self.train_tuple.dataset.label2ans)
        if args.load is not None: #fine-tuned weights from previous run
            self.load(args.load)
        ###Selectively freeze layers:
        # for rlayer in range(args.rlayers):
        #     self.model.lxrt_encoder.model.bert.encoder.r_layers[rlayer].attention.self.tempquery.weight.requires_grad=False
        #     self.model.lxrt_encoder.model.bert.encoder.r_layers[rlayer].attention.self.tempkey.weight.requires_grad=False
        #     self.model.lxrt_encoder.model.bert.encoder.r_layers[rlayer].attention.self.tempvalue.weight.requires_grad=False

        ###Otherwise, freeze everything and unfreeze specific layers
        # for param in self.model.parameters():
        #     param.requires_grad=False
            ###And selectively unfreeze layers after this


        #Testing process
        if args.test==True:
            args.img_num=1e12 #Always test on full test set
            args.dotrain=False
            self.testset=DecExpDataset(args.test)
            print("Number of samples in test set :", self.testset.__len__())

            #Perform evaluation only
            test_accuracy, avg_test_loss=decexp.evaluate(decexp.testset)
            print("Test accuracy", test_accuracy*100., "Avg loss: ", avg_test_loss)
            with open(args.output + '/test_acc_model_n_n_n', 'a') as f:
                    f.write("\n\nModel: ")
                    f.write(args.load)
                    f.write("\nTolerance: 0.5")
                    f.write("\nNumber of needed bits: 25")
                    f.write("\nHeads: 12")
                    f.write("\nTest accuracy: ")
                    f.write(str(test_accuracy*100.))
                    f.write(" Avg loss: ")
                    f.write(str(avg_test_loss))
                    f.flush()


        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        #Class weighting
        #if args.dotrain==True:
        #    print("Building class weights")
        #    temp_loader = DataLoader(
        #            self.trainset, batch_size=args.batch_size,
        #            shuffle=True, num_workers=args.num_workers,
        #            drop_last=False, pin_memory=False)
        #    nsamples1=np.zeros(ANSWER_LENGTH)
        #    #pdb.set_trace()
        #    temp_barloader=tqdm(temp_loader, total=int(args.img_num/args.batch_size))
        #    #print(next(iter(temp_loader)))
        #    for batchdata in temp_barloader:
        #        temp_labels = batchdata[5]
        #        for sample in temp_labels:
        #            for i in range(ANSWER_LENGTH):
        #                if sample[i]==1:
        #                    nsamples1[i]+=1
        #    del temp_loader, temp_labels #Freeing up memory
        #    print(nsamples1)
        #    clweight= torch.from_numpy((self.trainset.__len__()-nsamples1) / nsamples1)
        #    for i in range(ANSWER_LENGTH): #Preventing infinite values
        #        if clweight[i]>10000:
        #            clweight[i]=10000
        #    clweight=clweight.cuda()
        #    print(clweight)
        #else:
        #    clweight=torch.ones([ANSWER_LENGTH])

        # Loss and Optimizer
        print("batch_size:", args.batch_size)
        #self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=clweight, reduction='none') #Loss for multi-label multi-class classification. Choice: mean of the outputs
        #clweight=torch.ones([ANSWER_LENGTH])
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.batches_per_epoch=int(np.ceil(self.trainset.__len__()/args.batch_size))
        print("Batches per epoch: ", self.batches_per_epoch)
        t_total = int(self.batches_per_epoch * args.epochs)
        print("Total number of batches): %d" % t_total)

        if 'bert' in args.optim:
            self.optim = BertAdam(list(self.model.parameters()),
                                lr=args.lr,
                                warmup=0.1,
                                t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)


    def train(self):
        print("Starting training")

        with open("Query_list_pre.txt", 'r') as f:
            querylist=f.readlines()
            for i in range(len(querylist)):
                querylist[i]=querylist[i].replace("\n", "")

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
                idx, img_id, obj_num, feats, boxes, objs, labeldigits, label = batchdata
                
                #idx=idx.tolist()
                #____________________________________
                #For one batch
                self.model.train() #Tells the model that we are in train mode
                self.optim.zero_grad() #Reset all gradients to 0
                feats, boxes, objs, labeldigits = feats.cuda(), boxes.cuda(), objs.cuda(), labeldigits.cuda() #label is the logits after sigmoid
                
                self.sent=[]
                chosen_image=0
                for batch in range(args.batch_size):
                    temp=""
                    for i in range(nso):
                        temp+=obj_cats[int(random.choice(objs[batch][chosen_image]))]["name"]
                        temp+=" "
                    for i in range(nsq):
                        temp+=random.choice(querylist)
                        temp+=" "
                    temp=temp[:-1]
                    self.sent.append(temp)

                logit = self.model(img_id[0], feats, boxes, self.sent) #Does the model return logits or labels ? -> logits

                assert logit.size(dim=1) == labeldigits.size(dim=1) == ANSWER_LENGTH, 'output vector dimension does not fit the expected length (25)'
                #l2_norm = sum([p.pow(2.0).sum() for p in self.model.parameters() if p.requires_grad])
                loss_unreduced = self.bce_loss(logit, labeldigits)
                #for i in range(4):
                #    loss_unreduced[i]*=LOSSRATIODEC2EXP
                loss= torch.mean(loss_unreduced) #+ args.l2reg * l2_norm
                
                #print("loss :", loss.item(), "\n")
                #loss = loss * logit.size(1) #Removing the mean reduction of bce_loss to get the sum of all components
                                            #Uncomment if the loss is chosen without specifying the reduction (defaults to mean)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                for indx,l in zip(idx, labeldigits.cpu().numpy()):
                    idx2ans[indx]=l #Make sure that imgid is in quotes to be used in DecExpEvaluate. idx2ans = {idx1:label1, ..., idxn:labeln}
                temp_loss_hist.append(loss.item())
                #self.loss_hist[epoch]=loss.item() #Save the loss at the end of each epoch for later plotting along with the epoch number as key
                str_loss="loss: {0:.3f}".format(loss.item())
                barloader.set_description(str_loss)
                
                del idx, img_id, obj_num, feats, boxes, objs, labeldigits, label
            
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
            #decexp.save('epoch_%d'% (epoch)) #Saves the weights for each epoch for later (re-)evaluation

            train_accuracy, bitwise_train_accuracy, trainloss, train_trace, trainF1, actiontrain_acc=self.evaluate(self.trainset, self.sent) #Computes training accuracy
            log_str="\nTraining --- \n  Label-wise accuracy: Epoch {0}: {1} % \n  Bit-wise accuracy: Epoch {0}: {2} \n  Bit-wise F1-score: Epoch {0}: {3} \n".format(epoch, train_accuracy*100., bitwise_train_accuracy*100., trainF1)

            with open(args.output + "/traintrace.log", 'a') as f:
                f.write(str(epoch))
                f.write(": ")
                f.write(str(train_trace))
                f.write("\n")


            if args.valid != "" and args.valid != None:
                val_accuracy, bitwise_val_accuracy, val_loss, val_trace, valF1, actionval_acc=self.evaluate(self.validset, self.sent) #Compute validation accuracy
                self.valacc_hist[epoch]=val_accuracy #Save the validation accuracy from the current epoch for later plotting
                self.val_loss_hist[epoch]=val_loss
                self.bitwise_valacc_hist[epoch]=bitwise_val_accuracy
                self.valF1_hist[epoch]=valF1
                if val_accuracy > best_valid: #If the validation accuracy from this epoch is better than the current best from a previous epoch, save it
                    best_valid = val_accuracy
                    #self.save("BEST_val")
                if bitwise_val_accuracy > best_bitwise_valid:
                    best_bitwise_valid=bitwise_val_accuracy
                if valF1 > best_F1:
                    best_F1=valF1

                log_str += "Validation --- \n  Label-wise accuracy: \n    Epoch {0}: {1} %\n    Epoch {0}: Best {2} %\n  Bit-wise accuracy: \n    \
    Epoch {0}: {3} %\n    Epoch {0} Best Bit-wise accuracy {4} %\n  Bit-wise F1-score: \n    Epoch {0}: {5} \n    Epoch {0}\
 Best Bit-wise F1-score {6} \n".format(epoch, val_accuracy*100., best_valid*100., bitwise_val_accuracy*100., best_bitwise_valid*100., valF1, best_F1)
            
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
        batchval=0
        
        for data in tqdm(eval_loader, total=eval_batches):
            batchval+=1
            idx, img_id, obj_num, feats, boxes, objs = data[0:6]
            
            obj_cats=list(json.load(open("visual_genome_categories.json")).values())[0]
            sents=[]
            for batch in range(args.batch_size):
                temp=""
                for i in range(100):
                    temp+=obj_cats[int(objs[batch, 0, i])]["name"] #Deriving sents from the first image
                    temp+=" "
                temp+=(QUERY_LENGTH-2-len(temp.split()))*"[SEP] " #Padding
                temp=temp[:-1]
                sents.append(temp)

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
                for index, l in zip(idx, label.cpu().numpy()):
                    idx2anseval[index] = np.array(l)
            #if dump is not None:
            #    self.dump_result(imgid2ans, dump)
        return img_id, idx2anseval

    def evaluate(self, dset, sents, dump=None):
        """Evaluate all data in data_tuple."""
        print("\n Starting evaluation")
        
        self.val_loss={}
        #if args.img_num!=None:
        #    val_batches=int(np.ceil(args.img_num/args.batch_size))
        #else:
        eval_batches=int(np.ceil(dset.__len__()/args.batch_size))
        print("Number of evaluation batches:", eval_batches)
        eval_answers={}
        score = 0.
        tolerance=0.5 #We accept a confidence level of 1-0.5=0.5, that is 50% (Binary classification)
        nbequal=ANSWER_LENGTH #number of bits within each prediction that have to be equal to the label in order to classify as correct.
        eval_loader = DataLoader(
            dset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            drop_last=True, pin_memory=False)
        print("Dataloader loaded")
        imgids, temp=self.predict(eval_loader, eval_batches, sents, dump)
        eval_answers.update(temp)
        
        if args.save_predictions==True:
            with open(args.output + "/predictions.log", 'a') as f:
                for idx in eval_answers.keys():
                    f.write(str(dset.idx2label[idx][0]))
                    f.write(", ")
                    f.write(str(dset.idx2label[idx][1]))
                    f.write(", ")
                    f.write(str(eval_answers[idx]))
                    f.write("\n")
        
        #get labels for the idx chosen by the dataloader from the dataset annotations
        labels={}
        for indx in temp.keys():
            labels[indx]=(np.array(dset.idx2label[indx][1]))

        del temp
        trace=np.zeros(ANSWER_LENGTH) #Count position-wise errors over the dataset
        fulllabel_score=0
        actionlabel_score=0
        bitwise_score=0
        Falsepos=0
        Truepos=0
        Falseneg=0

        for indx in eval_answers.keys():
            temp=0
            tempactions=0
            
            for bit in range(4):
                if eval_answers[indx][bit]>=labels[indx][bit]-tolerance and eval_answers[indx][bit]<labels[indx][bit]+tolerance:
                    actionlabel_score+=1
            
            #if tempactions==4:
                #actionlabel_score+=1

            for bit in range(ANSWER_LENGTH):
                if eval_answers[indx][bit]>=labels[indx][bit]-tolerance and eval_answers[indx][bit]<labels[indx][bit]+tolerance:
                    bitwise_score+=1
                    temp+=1
                else:
                    trace[bit]+=1 #Keeps track of bits that are incorrectly predicted
                Falsepos+=int(eval_answers[indx][bit]+0.5)*(1-int(labels[indx][bit]))
                Truepos+=int(eval_answers[indx][bit]+0.5)*int(labels[indx][bit])
                Falseneg+=(1-int(eval_answers[indx][bit]+0.5))*int(labels[indx][bit])


            #if np.allclose(eval_answers[indx],labels[indx], rtol=0, atol=tolerance):
                #score+=1
            self.val_loss[indx] = nn.BCELoss()(torch.tensor(eval_answers[indx], dtype=torch.float32), 
                torch.tensor(labels[indx], dtype=torch.float32)).item()

            if temp>=nbequal: #If a minimum number of bits correspond between prediction and annotation, score +=1
                                #NOT THE SAME AS ADDING 1 TO SCORE IF A BIT CORRESPONDS (i.e computing the % of correct bits overall)
                fulllabel_score+=1

        print(trace)
        listlosses=list(self.val_loss.values())
        avg_eval_loss=sum(listlosses)/len(listlosses)
        actionlabel_acc=actionlabel_score/(len(eval_answers)*4)
        label_acc=fulllabel_score/len(eval_answers)
        bitwise_acc=bitwise_score/(len(eval_answers)*ANSWER_LENGTH)
        F1score=Truepos/(Truepos+0.5*(Falsepos+Falseneg))
        try:
            print("\nprecision: ", (Truepos/(Truepos+Falsepos)))
            print("\nrecall: ", (Truepos/(Truepos+Falseneg)))
            print("Bitwise action accuracy: ", actionlabel_acc*100.)
        except:
            print("error while computing precision, recall or accuracy")    
        return label_acc, bitwise_acc, avg_eval_loss, trace, F1score, actionlabel_acc


    def dump_result(self, imgid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = { 
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for imgid, ans in imgid2ans.items():
                ans=ans.tolist()
                print("imgid: ", type(imgid))
                print("answer: ", type(ans))
                result.append({
                    'img_id': imgid,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    def save(self, name):
        torch.save(self.model.state_dict(),
                os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
        print("Weights loaded")


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

if __name__ == "__main__":
    # Set process start-method
    # from torch.multiprocessing import set_start_method
    # try:
    #     set_start_method('spawn')
    # except:
    #     print("cannot set start method to 'spawn'")
    #     sys.exit()
    # Build Class
    args.img_num=100
    args.train='train'
    args.valid='val'
    args.test=None
    args.batch_size=2
    args.epochs=300
    #args.output='./snap/no_clweight_reg_0p0003_train_5000_12h_3_3_3_bs8_feats100'
    args.output='./snap/bdd100k/temp_to_delete'
    args.lr=5e-5
    args.num_decoderlayers=1
    
    #Load decexp fine-tuned model weights. Note: It is different from loading LXMERT pre-trained weights.
    #args.load='./snap/lastframe/x_varqueries_meanlvpool_train_all_12h_9_4_5_bs8/epoch_2' 
    
    args.load_lxmert='./snap/model' #load LXMERT pre-trained weights
    args.save_predictions=False
    args.save_heatmap=False
    args.num_workers=0
    
    args.dotrain=True #True: trains the model. False: Performs evaluation only
    args.heads=12
    args.llayers=1
    args.xlayers=1
    args.rlayers=1
    args.l2reg=0.
    nso=8
    nsq=QUERY_LENGTH-nso

    #Question
    #evalsents=["traffic light red yellow green lane forward stop left right obstacle car person rider sign other"]
    evalsents=["person truck headlight sky license plate bicycle windshield people child"]
    #evalsents=["table guitar tobacco pen screw keyboard laser magnet"]
    #evalsents=["building building flag flag flag flag building light light street people light street building building flag street car man window light window light people crosswalk street building sign light window building building man window sign window building street sign man flag flag window building window person man window taxi street light light street window building light light street flag window line sign street window window street window light car man car light building city window sign pole sign street window light window light crosswalk window awning person building light car window people building street building flag crosswalk street window traffic"]
    #evalsents=["building"]

    decexp = DecExp()

    if args.test!=True:
        if args.dotrain==False:
            #Perform evaluation only
            print("Performing evaluation only")
            #train_accuracy, avg_train_loss=decexp.evaluate(decexp.trainset)
            for i in range(1,2):
                with torch.no_grad():
                    print("loading epoch", i)
                    #decexp.load('./snap/train_full_3h_3_3_3_x/epoch_'+str(i))
                eval_label_accuracy, eval_bitwise_accuracy, avg_eval_loss, eval_trace, evalF1, action_acc=decexp.evaluate(decexp.validset, evalsents)
                #print("Train accuracy", train_accuracy*100., "Avg loss: ", avg_train_loss)
                print("Val accuracy", eval_label_accuracy*100., "Avg loss: ", avg_eval_loss)

                log_str = "Evaluation --- \n  Label-wise accuracy: \n    Epoch {0}: {1} %\n  Bit-wise accuracy: \n    Epoch {0}: {2} %\n\
  Bit-wise F1-score: \n    Epoch {0}: {3} \n".format(i, eval_label_accuracy*100., eval_bitwise_accuracy*100., evalF1)
                print(log_str)

                with open(args.output + "/eval_acc_loss.log", 'a') as f:
                        f.write("Model: ")
                        f.write(args.load)
                        #f.write("\nTrain accuracy: ")
                        #f.write(str(train_accuracy*100.))
                        #f.write(" Avg loss: ")
                        #f.write(str(avg_train_loss))
                        f.write("\nEval label accuracy: ")
                        f.write(str(eval_label_accuracy*100.))
                        f.write("\nEval bitwise accuracy: ")
                        f.write(str(eval_bitwise_accuracy*100.))
                        f.write("\nEval bitwise F1-score: ")
                        f.write(str(evalF1))
                        f.write("\nAction-wise accuracy ")
                        f.write(str(action_acc))
                        f.write("\nAvg loss: ")
                        f.write(str(avg_eval_loss))
                        f.write("\n\n")
                        f.flush()

                with open(args.output + "/evaltrace.log", 'a') as f:
                    f.write(str(i))
                    f.write(": ")
                    f.write(str(eval_trace))
                    f.write("\n")

        else: #Call the training function
            print("Splits in data: \n", decexp.trainset.split)
            if args.valid!=None and args.valid!="":
                print("", decexp.validset.split, "\n")
            decexp.train()
        
        
