import json
import numpy as np
from collections import Counter


path = './input/ISVQA/Annotation/imdb_nuscenes_trainval.json'
new_path = './input/ProcessedFile/imdb_nuscenes_trainval_add_score_id.json'

data = json.load(open(path))['data']
for i, datum in enumerate(data):
    datum['id']=i
    answers = datum['answers']
    label = dict(Counter(answers))
    for answer in label.keys():
        label[answer] = np.minimum(label[answer]/2, 1)
    datum['label'] = label

with open(new_path, 'w') as f:
    json.dump(data, f)

