import json
import os
import pdb

answers=json.load(open("../input/bdd100k/BDD100k_id2frameslabels.json", 'r'))
answers_updated={}
Nonelist=[None for i in range(4)] #Initialize list with all question types

##Create dict from scratch with all imgids
for imgid in answers.keys():
    answers_updated[imgid]={}
    for start_t in answers[imgid]:
        answers_updated[imgid].update({start_t:[None for i in range(4)]})
    
#print(answers_updated)
#pdb.set_trace()

#Add labels for whether a red traffic light exists in the video
red_traffic_light_keys=[]
for imgid in answers.keys():
    for start_t in answers[imgid].keys():
        if ("red" in answers[imgid][start_t][2] and "light" in answers[imgid][start_t][2]) or \
            ("red" in answers[imgid][start_t][3] and "light" in answers[imgid][start_t][3]):
                red_traffic_light_keys.append((imgid, start_t, answers[imgid][start_t][3]))

for tuple_ in red_traffic_light_keys:
    answers_updated[tuple_[0]][tuple_[1]][0]=1
    

#Add labels for whether a yellow traffic light exists in the video
yellow_traffic_light_keys=[]
for imgid in answers.keys():
    for start_t in answers[imgid].keys():
        if ("yellow" in answers[imgid][start_t][2] and "light" in answers[imgid][start_t][2]) or \
            ("yellow" in answers[imgid][start_t][3] and "light" in answers[imgid][start_t][3]):
                yellow_traffic_light_keys.append((imgid, start_t, answers[imgid][start_t][3]))

for tuple_ in yellow_traffic_light_keys:
    answers_updated[tuple_[0]][tuple_[1]][1]=1


#Add labels for whether a green traffic light exists in the video
green_traffic_light_keys=[]
for imgid in answers.keys():
    for start_t in answers[imgid].keys():
        if ("green" in answers[imgid][start_t][2] and "light" in answers[imgid][start_t][2]) or \
            ("green" in answers[imgid][start_t][3] and "light" in answers[imgid][start_t][3]):
                green_traffic_light_keys.append((imgid, start_t, answers[imgid][start_t][3]))

for tuple_ in green_traffic_light_keys:
    answers_updated[tuple_[0]][tuple_[1]][2]=1

pdb.set_trace()
