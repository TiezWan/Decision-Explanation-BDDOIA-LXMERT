# LXMERT based ISVQA in autonomous driving dataset (Nuscenes)
This repository presents an LXMERT-based model performing VQA-based scene understanding for autonomous driving (dataset: Berkeley DeepDrive - eXplanation (BDD-X)).

Instructions on how to use the code, and train/test models are found at the bottom of this file.

## Dataset introduction
```sh
|-- input
|   |-- bdd100k
|       |[ToDo] Sort video data into the proper folders
|       |-- trainclips_downsampled_Xfpv
|       |-- valclips_downsampled_Xfpv
|       |-- testclips_downsampled_Xfpv
|       |-- feature_output
|           |-- trainclips_downsampled_Xfpv
|           |-- valclips_downsampled_Xfpv
|-- src
|    |-- lxrt (deprecated)
|    |-- mmf
|    |-- pretrain (duplicate)
|    |-- vqa-maskrcnn-benchmark
|-- src_DecExp
|    |-- lxrt
|    |-- pretrain
|    |--snap
|    |-- DecExp.py
|    |-- ...
```

## File structure
```sh
[ToDo] This is not correct
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
Make sure to create a new conda environment before proceeding, do not perform the following operations on the base environment as this sometimes leads to troubles.

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
The question being sent is currentoy being coded in DecExp.py, under the "self.sents" variable. One can either send a question to the model or let the model figure out a query question (a collection of words) from the objects detected in the data.

The annotation file for the BDD-X videos is accessible under the input folder of this repository.

# Feature extraction
Make sure to modify the output path in feature_extraction_DecExp.py and run the following line to extract features from video data.
```sh
python feature_extraction_DecExp.py
```

# Training and Testing
The video data can be obtained through the dedicated BDD web page. All the splits in the BDD-X are sub-splits of the train split in the BDD data. This means that here is no need to download the test and val split from the BDD website. The downloaded videos then need to be split into clips of varying lengths.

To run the code, make sure that your previously setup conda env is running and selected.

Training or testing is decided in the Decexp.py file by modifying the args.train and args.valid variables. 

To perform training, input args.train='train'. Optionally, by setting args.valid='val', one can get an evaluation on the val split for each epoch (concurrently with the training)

To perform testing only, set args.train=None and args.valid='test' (though it can be done on any split)

[ToDo] Add test split, modify code to use args.test

Modify other parameters such as the number of training samples per epoch, the learning rate, etc... and run the following command to start training/testing:

```sh
python DecExp.py
```

# Result
The results are stored in the src_DecExp/snap/$model_name folder. In the current version, saving the model's weights at each epoch has been disabled to prevent excessive and unnecessary memory usage. One can simply re-enable it by uncommenting the corresponding line in DecExp.py

