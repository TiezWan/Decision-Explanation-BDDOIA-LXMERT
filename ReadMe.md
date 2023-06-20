# Scene Understanding for Autonomous Driving Using Visual Question Answering

This repository presents an LXMERT-based model performing VQA-based scene understanding for autonomous driving, using Berkeley Deep Drive-X (eXplanation) Dataset.

## File structure
```sh
|-- src
|   |-- lxrt_base
|   |   |-- entry.py
|   |   |-- file_utils.py
|   |   |-- modeling.py
|   |   |-- optimization.py
|   |   |-- tokenization.py
|   |-- lxrt_modified
|   |   |-- entry.py
|   |   |-- file_utils.py
|   |   |-- modeling.py
|   |   |-- optimization.py
|   |   |-- tokenization.py
|   |-- utils
|   |   |-- logger.py
|   |   |-- param.py
|   |   |-- utils.py
|   |-- dataset.py
|   |-- model.py
|   |-- optimizer.py
|-- feature_extaction.py
|-- main.py
|-- ReadMe.md
```

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