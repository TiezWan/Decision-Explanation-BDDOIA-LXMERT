from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

f=open('snap/no_clweight_reg_0p0003_train_5000_12h_3_3_3_bs8/traintrace.log', 'r')
trainlist=[]
for i in range(22):
    trainlist.append(f.readline()[5:-2].split(', '))
    for j in range(25):
        trainlist[i][j]=int(trainlist[i][j])

f=open('snap/no_clweight_reg_0p0003_train_5000_12h_3_3_3_bs8/valtrace.log', 'r')
vallist=[]
for i in range(22):
    vallist.append(f.readline()[5:-2].split(', '))
    for j in range(25):
        vallist[i][j]=int(vallist[i][j])

epochs=[i for i in range(1,23)]
classes=[i for i in range(1,26)]

plt.bar(classes, trainlist[0])
plt.show()