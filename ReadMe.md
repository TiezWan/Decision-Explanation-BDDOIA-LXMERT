# Scene Understanding for Autonomous Driving Using Visual Question Answering

TThis project explores the dot products of self-attention mechanisms as an innovative approach to enhance the explainability of autonomous driving systems. We implement a Visual Question Answering framework with two models: Modeling_base.py (standard LXMERT) and Modeling_modified.py (modified LXMERT). These models are evaluated for recognizing road signs and traffic lights, by using Berkeley Deep Drive-X (eXplanation) Dataset.

<!-- ToDo in general:
1. use black to format the codes
2. complete the readme with the template: https://github.com/equinor/data-science-template/blob/master/README.md
3. add the paper information and the architecture into the readme. ref: https://github.com/SHI-Labs/OneFormer
-->

## File structure
```sh
|-- input -> a softlink to VQA_input
|-- checkpoints
|   |-- Weights_modified.pth
|   |-- Weights_baseline.pth
|-- bdd100k_image
|   |-- train
|   |-- val
|   |-- test
|-- annotation
|   |-- questionset.json
|   |-- trainlabels.json
|   |-- vallabels.json
|   |-- testlabels.json
|   |-- testlabels_heatmap.json
|   |-- greenlight.json
|   |-- redlight.json
|   |-- roadsigns.json
|-- extracted_features
|   |-- train
|   |-- val
|   |-- test
|
|-- src
|   |-- lxrt
|   |   |-- entry.py
|   |   |-- file_utils.py
|   |   |-- modeling_base.py
|   |   |-- modeling_modified.py
|   |   |-- optimization.py
|   |   |-- tokenization.py
|   |-- utils
|   |   |-- logger.py
|   |   |-- param.py
|   |   |-- utils.py
|   |-- dataset.py
|   |-- model.py
|   |-- optimizer.py
|-- |-- feature_extaction.py
|
|-- .gitignore
|-- main.py
|-- ReadMe.md
|-- requirements.txt

## Preparation
For environment configuration, dataset generation, prediction & heatmap visualization, please follow: [ToDo]

This repository only shows the rearranged code associated with training and testing process. 

## Training
Instead of training the model from scratch, the pretrained weights from LXMERT model are used. To fine-tune the LXMERT model using Berkeley Deep Drive-X (eXplanation) Dataset as baseline, run:
```sh
python DecExp.py
```

Since our modified LXMERT model do not have additional parameters, we can also utilize the pretrained weights and fine-tune the modified LXMERT model by running:
```sh
python DecExp.py
```

The fine-tuned weights will be saved at `output`.

## Testing
Load the fine-tuned weights after training. Similar to training, one can use either the original LXMERT model as baseline or the modified LXMERT model to perform testing.

For the original LXMERT model, run:
```sh
# ToDo: python DecExp.py
```

For the modified LXMERT model, run:
```sh
# ToDo: python DecExp.py
```