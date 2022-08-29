from DecExp import DecExp
from param import args
import numpy as np
import sys
import os

IMG_NUM=args.img_num
args.train='train'
args.valid='val'
args.batch_size=16
args.epochs=1 #number of successive epochs for each image batch
args.tiny=True
args.dotrain=False
dir='./snap/train_full_12h_1_1_1/'
args.dotrain=True
args.llayers=1
args.xlayers=1
args.rlayers=1

for modelnb in range(0,10):
    print("Evaluating model n°", modelnb+1)
    args.load=dir+'epoch_'+str(modelnb+1)
    decexp = DecExp()

    #Perform evaluation only
    train_accuracy, avg_train_loss=decexp.evaluate(decexp.trainset)
    val_accuracy, avg_val_loss=decexp.evaluate(decexp.validset)
    print("Train accuracy", train_accuracy*100., "Avg loss: ", avg_train_loss)
    print("Val accuracy", val_accuracy*100., "Avg loss: ", avg_val_loss)
    with open(dir + '/acc_loss_'+dir[18:-1]+'.log', 'a') as f:
                    f.write("\n\nModel: ")
                    f.write(args.load)
                    f.write("\nTolerance: 0.5")
                    f.write("\nNumber of needed bits: 25")
                    f.write("\nHeads: 12")
                    f.write("\nTrain accuracy: ")
                    f.write(str(train_accuracy*100.))
                    f.write(" Avg loss: ")
                    f.write(str(avg_train_loss))
                    f.write("\nVal accuracy: ")
                    f.write(str(val_accuracy*100.))
                    f.write(" Avg loss: ")
                    f.write(str(avg_val_loss))
                    f.flush()