# LXMERT based ISVQA in autonomous driving dataset (Nuscenes)
This repository is to implement lxmert model based VQA in autonomous driving dataset(Nuscenes) 

## Dataset introduction
[ToDo] blabla
```sh
|-- extracted_features
|   |-- train
|   |-- test
|-- mini
|   |-- maps
|   |   |-- 36092f0b03a857c6a3403e25b4b7aab3.png
|   |-- samples
|   |   |-- CAM_BACK
|   |   `-- ....
|   `-- v1.0-mini
|       |-- attribute.json
|       `-- ...
|-- part01
|   `-- samples
|       |-- CAM_BACK
|       `-- ...
|-- part02
|   `-- samples
|       |-- CAM_BACK
```

## File structure
```sh
-- input
  -- ISVQA
    -- NuScenes
      -- extracted_features
    -- jsons
    -- pretrained
      --pretrained weights
  -- ProcessedFile
-- output
-- others
-- src
|   -- DataPreproScript
|   -- lxrt
|   -- pretrain
|   -- param.py
|   -- vqa_data_preprocessing.py
|   -- vqa_model.py
-- feature_extaction.py
-- ISVQA_main.py
-- ReadMe.md
```


## Preparation
### Prerequisite
conda install -c anaconda boto3  # TODO
- MMF install [Official instruction](https://mmf.sh/docs/), download the mmf repo under '/src'.
```
cd src
git clone https://github.com/facebookresearch/mmf.git
cd mmf
pip install --editable .  # after this one, it will become pytorch 1.9 automatically
```
- Mask-RCNN backbone [instruction](https://mmf.sh/docs/tutorials/image_feature_extraction/), download the repo under '/src'.
```
pip install ninja yacs cython matplotlib
pip install opencv-python
cd src
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build develop
```
- You might have such a following bug, change PY3 to PY37:
File "/root/Documents/ISVQA/src/vqa-maskrcnn-benchmark/maskrcnn_benchmark/utils/imports.py", line 4, in <module>
    if torch._six.PY3:
AttributeError: module 'torch._six' has no attribute 'PY3'
```
- LXMERT repository [instruction](https://github.com/airsplay/lxmert/blob/master/requirements.txt) 
- download pretrained lxmert model via
```sh
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```
- download *all_ans.json* for pretained lxmert model(https://github.com/airsplay/lxmert/blob/master/data/lxmert/all_ans.json)

- maskrcnn_benchmark
- mmf (orinially is ..., mmf is too large ...)
- cv
- python version, pytorch version


# Question ID and score generation
For original question id is not unique for each question, we need to generate new question id to identify each question.
Besides, we also need to generate answer score for each answer.

You need to preprocess the original annotation files on your own following below steps.
1. Download the original annotation files from [ISVQA](https://github.com/ankanbansal/ISVQA-Dataset/tree/master/nuscenes)
2. Generate new annotation file and answer file via *add_id_and_score.py* under *input/others*. Before running *add_id_and_score.py*, don't forget to change file path to yours.

# Feature extraction
```sh
python feature_extraction.py
```

# Training and Test
You need to split the original trainpart features into training and test part features, so you can generate two new annotation files as indicators, including train part and test part, via *spiltjson.py* under *input/others*. 
When two new .json file are ready, run *ISVQA_main.py* to train and test the whole model.
```sh
python ISVQA_main.py
```

# Result
After 30 Epochs, we have the accuracy on training set as xxx and on test set as xxx.

*figure1*

*figure2*

*figure3*

