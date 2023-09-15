import os, json, random, logging, pdb
import numpy as np
import torch
import torch.utils.data
from torch import Tensor
from typing import Tuple

from src.utils.param import args

logger = logging.getLogger(__name__)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_idx: str) -> None:

        self.file_idx = file_idx
        logger.info(f"Loading dataset for {self.file_idx}...")

        # Get raw data
        if self.file_idx == "train" or self.file_idx == "val":
            avail_data = os.listdir(f"{args.input}/{self.file_idx}clips_downsampled_6fpv_2sampled/")
            raw_data = json.load(open(f"{args.input}/BDD100k_subtasks_{self.file_idx}labels_new.json"))
        elif self.file_idx == "test":
            if args.save_heatmap:
                avail_data = os.listdir(f"{args.input}/{self.file_idx}clips_heatmap_images/")
                raw_data = json.load(open(f"{args.input}/BDD100k_subtasks_{self.file_idx}labels_heatmap.json"))
            else:
                avail_data = os.listdir(f"{args.input}/{self.file_idx}clips_downsampled_6fpv_2sampled/")
                raw_data = json.load(open(f"{args.input}/BDD100k_subtasks_{self.file_idx}labels_new.json"))                
        else:
            raise ValueError(f"Unknown evaluation type: {self.file_idx}")

        # Get questions
        self.queries = json.load(open(f"{args.input}/questionSet.json", 'r'))

        self._preprocess_data(raw_data)
        self._check_samples_num(len(avail_data))

        logger.info(f"Loading dataset for {self.file_idx}...Done! Dataset with {self.__len__()} samples.")

    def __len__(self) -> int:
        """Give the length of preprocessed data"""

        return len(self.idx2label)

    def __getitem__(self, idx: int):
        """Fetch the data for one frame image"""

        # Load image features
        try:
            img_id = self.idx2label[idx][0]
            img_data = np.load(f"{args.input}/feature_output/{self.file_idx}clips_downsampled_6fpv_2sampled/{img_id}.npy", allow_pickle=True)[0]
        except:
            logger.debug(f"Error while fetching sample {img_id}.")
            pdb.set_trace()
       
        # Fetch all the variables
        img_id = img_id
        boxes = self._fetch_boxes(img_data)
        features = torch.tensor(img_data['features'], dtype=torch.float32)
        obj_num = img_data['num_boxes']
        objects = img_data['objects']
        
        random.seed(args.seed)
        ques_types = list(self.queries.keys())
        avail_ques_idx = (0, 2, 4)  # 0: red lights; 2: green lights; 4: road signs

        if self.file_idx == "train" or self.file_idx == "val":
            ques_idx = random.choice(avail_ques_idx)
            # Append possibility of the Counter question types: 1, 3, 5 
            ques_appendix = random.choice((0, 1))
            sent = random.choice(self.queries[ques_types[ques_idx + ques_appendix]])
            label = torch.tensor(self.idx2label[idx][1][ques_idx + ques_appendix], dtype=torch.float32)  
        # There are two datasets for testing, normal dataset is just for testing, another one is for heatmap visulization          
        elif self.file_idx == "test":
            if args.save_heatmap:
                sent = [] # 3 questions for heatmap testset
                label = []
                for ques_idx in avail_ques_idx:
                # Append possibility of the Counter question types: 1, 3, 5 
                    ques_appendix = random.choice((0, 1))
                    sent_content = random.choice(self.queries[ques_types[ques_idx + ques_appendix]])
                    sent.append(sent_content)
                    label = np.append(label, self.idx2label[idx][1][ques_idx + ques_appendix])
                    
                label = torch.tensor(label, dtype=torch.float32)
            else:
                ques_idx = random.choice(avail_ques_idx)
                # Append possibility of the Counter question types: 1, 3, 5 
                ques_appendix = random.choice((0, 1))
                sent = random.choice(self.queries[ques_types[ques_idx + ques_appendix]])
                label = torch.tensor(self.idx2label[idx][1][ques_idx + ques_appendix], dtype=torch.float32)     
        else:
            raise ValueError(f"Unknown evaluation type: {self.file_idx}")

        assert obj_num == objects.shape[0] == len(boxes) == len(features), f"Size didn't match: obj_num = {obj_num}, objects.shape = {objects.shape}, len(boxes) = {len(boxes)}, len(features) = {len(features)}"

        return idx, img_id, boxes, features, sent, label
 
    def _preprocess_data(self, raw_data: dict) -> None:
        """Preprocess the raw data"""

        # key: index; 1st value: image ID (str); 2nd value: label in this image (np.array)
        self.idx2label = {}

        i = 0
        for clip_id in raw_data.keys():
            for (frame_key, frame_label) in raw_data[clip_id].items():
                if any(frame_label):
                    # If the label has at least one non-None element
                    frame_label = np.array(frame_label, dtype=float)
                    self.idx2label[i] = (frame_key.split(".")[0], frame_label)
                    i += 1
        
    def _check_samples_num(self, data_length: int) -> None:
        """Update args.samples_num after fetching the data"""

        if (args.samples_num is None or args.samples_num > data_length):
            args.samples_num = len(self.idx2label)
        else:
            # For debug
            self.idx2label = {idx: self.idx2label[idx] for idx in range(args.samples_num)}
            logger.info(f"The number of samples was limited for debugging.")

    def _fetch_boxes(self, img_data) -> Tensor:
        """Fetch the boxes and normalize it"""

        # Normalizing boxes to [0,1], mode xyxy
        boxes_pre = img_data['bbox'].copy()
        boxes_pre[:, (0, 2)] /= 1289.6024
        boxes_pre[:, (1, 3)] /= 736.50415
        np.testing.assert_array_less(boxes_pre, 1+1e-5, verbose=True)
        np.testing.assert_array_less(-boxes_pre, 0+1e-5, verbose=True)
        return torch.tensor(boxes_pre, dtype=torch.float32)


