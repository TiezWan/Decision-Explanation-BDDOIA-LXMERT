# Copyright (c) Facebook, Inc. and its affiliates.

# Requires vqa-maskrcnn-benchmark (https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark)
# to be built and installed. Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
# When the --background flag is set, the index saved with key "objects" in
# info_list will be +1 of the Visual Genome category mapping above and 0
# is the background class. When the --background flag is not set, the
# index saved with key "objects" in info list will match the Visual Genome
# category mapping.
import argparse
import os
import json
import sys

#sys.path.insert(0,"../vqa-maskrcnn-benchmark/")

import cv2 
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from mmf.utils.download import download
from PIL import Image

#torch.cuda.set_device('cuda:4')
#print([torch.cuda.device_ids])

class FeatureExtractor:
    MODEL_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
                 + "detectron_model/detectron_model.pth",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
                 + "detectron_model/detectron_model_x152.pth",
    }
    CONFIG_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
                 + "detectron_model/detectron_model.yaml",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
                 + "detectron_model/detectron_model_x152.yaml",
    }

    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()

        self.data = [json.load(open(self.args.annotations_file1)), json.load(open(self.args.annotations_file2)), json.load(open(self.args.annotations_file3))]

        self.parts = [os.listdir(self.args.image_dir1), os.listdir(self.args.image_dir2), os.listdir(self.args.image_dir3)]
        self.image_paths = [['input/data/train/'+ p for p in self.parts[0]]]
        self.image_paths.append(['input/data/val/'+ p for p in self.parts[1]])
        self.image_paths.append(['input/data/test/'+ p for p in self.parts[2]])
        for set in range(len(self.image_paths)):
            for image in self.image_paths[set]:
                assert os.path.isfile(image)
        #self.image_dir = [[paths to train image 1, path to train image 2, ...], [paths to val image 1, path to val image 2, ...],[paths to test image 1, path to test image 2, ...]]
        self._try_downloading_necessities(self.args.model_name) #same
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def _try_downloading_necessities(self, model_name):
        if self.args.model_file is None and model_name is not None:
            model_url = self.MODEL_URL[model_name]
            config_url = self.CONFIG_URL[model_name]
            self.args.model_file = model_url.split("/")[-1] #detectron_model_x152.pth
            self.args.config_file = config_url.split("/")[-1] #detectron_model_x152.yaml
            if os.path.exists(self.args.model_file) and os.path.exists(
                    self.args.config_file
            ):
                print(f"model and config file exists in directory: {os.getcwd()}")
                return
            print("Downloading model and configuration")
            download(model_url, ".", self.args.model_file)
            download(config_url, ".", self.args.config_file)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name", default="X-152", type=str, help="Model to use for detection"
        )
        parser.add_argument(
            "--model_file",
            default=None,
            type=str,
            help="Detectron model file. This overrides the model_name param.",
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="input/feature_output_2", help="Output folder"
        )
        parser.add_argument("--image_dir1", type=str, default="input/data/train/", help="Image directory or file")
        parser.add_argument("--image_dir2", type=str, default="input/data/val/", help="Image directory or file")
        parser.add_argument("--image_dir3", type=str, default="input/data/test/", help="Image directory or file")
        parser.add_argument("--annotations_file1", default="input/gt_4a_21r_train.json", type=str)
        parser.add_argument("--annotations_file2", default="input/gt_4a_21r_val.json", type=str)
        parser.add_argument("--annotations_file3", default="input/gt_4a_21r_test.json", type=str)
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
                 + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        print("Building detection model \n")
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))
        load_state_dict(model, checkpoint.pop("model"))
        model.to("cuda")
        model.eval()  
        print("model built successfully \n")
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:  
            im = np.array(img.convert("RGB")).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im = im[:, :, ::-1]  
        im -= np.array([102.9801, 115.9465, 122.7717]) 
        im_shape = im.shape  
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])  
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:  
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
            self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]  
        feats = output[0][feature_name].split(n_boxes_per_image)

        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros(scores.shape[0]).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater
                    # than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][:, start_index:], dim=1)
            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "cls_prob": scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []
        i=0
        totalim=len(image_paths)
        for image_path in image_paths:
            i+=1
            print("image n°", i, "/", totalim, "\n")
            im, im_scale, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)  
        current_img_list = current_img_list.to("cuda") 
        # current_img_list = current_img_list.to(device)

        with torch.no_grad():  
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )
        return feat_list

    def _save_feature(self, file_name, feature, info, output_dir):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        file_base_name = file_base_name + ".npy"
        feature.update(info)
        np.save(os.path.join(output_dir, file_base_name), feature)

    def extract_features(self):
        print("Starting feature extraction \n")
        for set in range(2,3): #For train, val and test set, repeat:
            print("Set n°", set, "\n")
            if set==0:
                output_dir="../../dataset/BDDOIA/feature_output/train/"
            elif set==1:
                output_dir="input/feature_output/val/"
            elif set==2:
                output_dir="input/feature_output/test/"
            #self.parts[set]
            #iterdata=iter(self.data[set].items()) #Load data for each set
            #for idx in range(len(self.data)):
                #currdata=next(iterdata) #Get current data sample
                #image_id=currdata[0] #Replace the id with the name 
                
                #image_ids.append(image_id) #if the current image wasn't read before, add it to the list of read images
            nbsplits=5831
            nbimages=len(self.image_paths[set])
            splitlen=int((nbimages/nbsplits))

            splits=[self.image_paths[set][(i*splitlen):((i+1)*splitlen)] for i in range(nbsplits)]

            for i in range(nbsplits):
                print("split n°", i, "\n")
                features, infos = self.get_detectron_features(splits[i])

                            # print(features.is_cuda, infos.is_cuda)
                k=0
                for info in infos:
                    info['image_id'] = self.parts[set][i*splitlen+k][:-4]
                    k+=1

                #features_new = {}
                features_list = []
                #features_all = []
                m=0
                for feature in features:
                    features_new = {}
                    features_new['image_id'] = self.parts[set][i*splitlen+m][:-4]
                                #features_all.append(feature)
                    features_new['feature'] = feature.cpu()
                                # print(features_new)
                    features_list.append(features_new)
                            #print(features_list)
                    m+=1

                        
                print('extract')
                for l in range(splitlen):
                    self._save_feature(splits[i][l][:-3]+'npy', features_list[l], infos[l], output_dir)

                
        # else:
        #
        #     files = get_image_files(
        #         self.args.image_dir,
        #         exclude_list=self.args.exclude_list,
        #         start_index=self.args.start_index,
        #         end_index=self.args.end_index,
        #         output_folder=self.args.output_folder,
        #     )
        #
        #     finished = 0
        #     total = len(files)
        #
        #     for chunk, begin_idx in chunks(files, self.args.batch_size):
        #         features, infos = self.get_detectron_features(chunk)
        #         for idx, file_name in enumerate(chunk):
        #             self._save_feature(file_name, features[idx], infos[idx])
        #         finished += len(chunk)
        #
        #         if finished % 200 == 0:
        #             print(f"Processed {finished}/{total}")


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
