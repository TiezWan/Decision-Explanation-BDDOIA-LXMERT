import json
import pdb

sentences=[]
rejected_ids=[]
id2frameslabels_raw=json.load(open("../input/bdd100k/BDD100k_id2frameslabels.json"))
for img_id in id2frameslabels_raw.keys():
    curr_img=id2frameslabels_raw[img_id]
    for start_time in curr_img.keys():
        try:
            sentences.append(curr_img[start_time][2])
            sentences.append(curr_img[start_time][3])
        except:
            print(f"Couldn't fetch the action or the explanation for img {img_id}, start time {start_time}")
            rejected_ids.append(f"{img_id}_{start_time}")
str_all=""

for sentence in sentences:
    str_all+=sentence
    str_all+=" "

str_all=str_all.replace(".", "").replace("(", ""). replace(")", "")

words=str_all.split(" ")

wordset=set(words)
wordset.add(".")
wordset.add("<sos>")
wordset.add("<eos>")
wordset.discard("")
print(len(wordset))

str_wordset=""
for word in wordset:
    str_wordset+=word
    str_wordset+=" "
pdb.set_trace()