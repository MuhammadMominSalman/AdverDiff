import json
import cv2
import numpy as np
import os.path
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import pandas as pd
import numpy as np
import torch
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
class BDD_10K_Dataset(Dataset):
    """Dataset class for the BDD_10K dataset."""

    def __init__(self, data_dir, mode, det=None, seed=1, preprocess= "seg"):
        """Initialize and preprocess the BDD_10K dataset."""
        # self.init_json_labels(data_dir, mode)
        self.mode = mode
        self.seed = seed
        self.preprocess= preprocess
        if mode == "train":
            if os.path.isfile(os.path.join(data_dir, "labels", "df_sampled.csv")):
                self.df_sampled = pd.read_csv(os.path.join(data_dir, "labels", "df_sampled.csv"))
            if os.path.isfile(os.path.join(data_dir, "labels", "train_df.csv")):
                self.df = pd.read_csv(os.path.join(data_dir, "labels", "train_df.csv"))
            else:
                self.json_to_df(data_dir, mode)
            self.image_dir = data_dir + "/train/"
        else:
            if os.path.isfile(os.path.join(data_dir, "labels", "df_sampled_val.csv")):
                self.df_sampled = pd.read_csv(os.path.join(data_dir, "labels", "df_sampled_val.csv"))
            if os.path.isfile(os.path.join(data_dir, "labels", "val_df.csv")):
                self.df = pd.read_csv(os.path.join(data_dir, "labels", "val_df.csv"))
            else:
                self.json_to_df(data_dir, mode)
                self.df.to_csv(os.path.join(data_dir, "labels", "val_df.csv"), index=False)
            self.image_dir = data_dir + "/val/"
        self.df = self.df[["name", "timeofday", "weather"]]
        
        # Data curation
        self.df = self.df[self.df["weather"] != "undefined"]
        self.df = self.df[self.df["timeofday"] != "undefined"]
        self.df = self.df.drop_duplicates(subset=['name'], keep='first')
        self.df["timeofday"]= self.df.apply(lambda x: x['timeofday'].replace('daytime','day'), axis=1)
        self.df["timeofday"]= self.df.apply(lambda x: x['timeofday'].replace('dawn/dusk','dawn'), axis=1)
        
        if self.preprocess == "seg":
            # OFCOCO segmentor # CAN Change
            self.det = "Seg_OFCOCO"
            # Get segmentation preprocessor
            if self.det == 'Seg_OFCOCO':
                self.preprocessor = OneformerCOCODetector()
            if self.det == 'Seg_OFADE20K':
                self.preprocessor = OneformerADE20kDetector()
            if self.det == 'Seg_UFADE20K':
                self.preprocessor = UniformerDetector()
            # if not os.path.isdir(os.path.join(data_dir, "seg")):
            self.preprocess_seg(data_dir)
            self.seg_dir = data_dir + "/seg/"
        elif self.preprocess == "ip2p":
            self.preprocessor = None
        elif self.preprocess == "canny":
            self.preprocessor = CannyDetector()
            self.preprocess_edge(data_dir)
            self.edge_dir = data_dir + "/edge/"

    def preprocess_seg(self, data_dir):
        for index, _ in self.df_sampled.iterrows():
            filename = self.df_sampled.iloc[index]["name"]
            if not os.path.isfile(os.path.join(data_dir, "seg", filename)):
                source = cv2.imread(os.path.join(self.image_dir, filename))
                # Do not forget that OpenCV read images in BGR order.
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                # Resize the images to 512x512
                source = cv2.resize(source, (512, 512), 
                    interpolation = cv2.INTER_LINEAR)

                # Get segmentation
                with torch.no_grad():
                    input_image = HWC3(source)
                    detected_map = self.preprocessor(resize_image(input_image, 512))
                    detected_map = HWC3(detected_map)

                    detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(data_dir, "seg", filename), detected_map)
    
    def preprocess_edge(self, data_dir):
        for index, _ in self.df_sampled.iterrows():
            filename = self.df_sampled.iloc[index]["name"]
            if not os.path.isfile(os.path.join(data_dir, "edge", filename)):
                source = cv2.imread(os.path.join(self.image_dir, filename))
                # Do not forget that OpenCV read images in BGR order.
                source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                # Resize the images to 512x512
                source = cv2.resize(source, (512, 512), 
                    interpolation = cv2.INTER_LINEAR)

                # Get edges
                with torch.no_grad():
                    input_image = HWC3(source)
                    detected_map = self.preprocessor(resize_image(input_image, 512), 100, 200)
                    detected_map = HWC3(detected_map)

                    detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(data_dir, "edge", filename), detected_map)
    
    def __len__(self):
        """Return the number of images."""
        return len(self.df_sampled)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename = self.df_sampled.iloc[index]["name"]
        weather = self.df_sampled.iloc[index]["weather"]
        tod = self.df_sampled.iloc[index]["timeofday"]

        source = cv2.imread(os.path.join(self.image_dir, filename))
        target = cv2.imread(os.path.join(self.image_dir, filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize the images to 512x512
        source = cv2.resize(source, (512, 512), 
               interpolation = cv2.INTER_LINEAR)
        
        target = cv2.resize(target, (512, 512), 
               interpolation = cv2.INTER_LINEAR)

        # Get segmentation
        if self.preprocess == 'seg' and os.path.isfile(os.path.join(self.seg_dir, filename)):
            control = torch.from_numpy(cv2.imread(os.path.join(self.seg_dir, filename))).float().cuda() / 255.0
        elif self.preprocess == 'edge' and os.path.isfile(os.path.join(self.edge_dir, filename)):
            control = torch.from_numpy(cv2.imread(os.path.join(self.seg_dir, filename))).float().cuda() / 255.0
        else:
            with torch.no_grad():
                input_image = HWC3(source)

                if self.preprocess == 'ip2p':
                    detected_map = input_image.copy()
                elif self.preprocess == 'canny':
                    detected_map = self.preprocessor(resize_image(input_image, 512), 100, 200)
                    detected_map = HWC3(detected_map)
                else:
                    detected_map = self.preprocessor(resize_image(input_image, 512))
                    detected_map = HWC3(detected_map)

                detected_map = cv2.resize(detected_map, (512, 512), interpolation=cv2.INTER_LINEAR)

                control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        
        prompt = "A {w} image showing a car driving down a road on a {w} {tod}.".format(w=weather, tod=tod)

        return dict(jpg=target, txt=prompt, hint=control)


    def init_json_labels(self, data_dir, mode):
        """Return one image and its corresponding attribute label."""
        # TODO: improve runtime - use pickle to save structured data?
        # parse json labels to a nice format
        # output: dictionary that maps img_name to it's json object
        json_path = os.path.join(data_dir, "labels")
        if mode =="train":
            file = open(os.path.join(json_path,"det_train.json"))
        else:
            file = open(os.path.join(json_path,"det_val.json"))
        data = json.loads(file.read())
        json_dict = {}
        for json_obj in data:
            key = json_obj['name'].split(".")[0]
            json_dict[key] = json_obj
        if mode == "train":
            self.train_json = json_dict
        else:
            self.val_json = json_dict
        return

    def json_to_df(self, data_dir, mode):
        json_path = os.path.join(data_dir, "labels")
        if mode == "train":
            with open(os.path.join(json_path,"det_train.json"),"r") as fh:
                labels_json = json.loads(fh.read())
            print("opened the file")
        else:
            with open(os.path.join(json_path,"det_val.json"),"r") as fh:
                labels_json = json.loads(fh.read())
            print("opened the file")

        # fix missing labels for train ds
        if mode == 'train':
            for i in range(len(labels_json)):
                json_obj = labels_json[i]
                if 'labels' not in json_obj.keys():
                    print(f"object {i}: {json_obj['name']} is missing labels!")
                    json_obj['labels'] = [{}]

        meta_list = ['name','attributes']
        df = pd.json_normalize(labels_json, record_path=['labels'], meta = meta_list, errors='ignore')
        print(f"generated base df. total number of images: {len(df.name.unique())}")
        new_cols = pd.json_normalize(df.attributes)
        print(f"generated att df.")
        data_df = pd.concat([df.drop(columns=['id','attributes']),new_cols], axis=1)
        print(f"merged dfs. total number of images: {len(data_df.name.unique())}")

        # rename + reorder columns
        data_df.columns = ['category', 'occluded', 'truncated', 'trafficLightColor', 'x1', 'y1', 'x2', 'y2', 'crowd', 'name', 'weather', 'timeofday','scene']
        data_df = data_df[['name', 'category', 'weather', 'timeofday','scene', 'occluded', 'truncated', 'crowd', 'trafficLightColor', 'x1', 'y1', 'x2', 'y2']]
        self.df = data_df
        return