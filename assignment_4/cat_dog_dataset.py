import os
import glob
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image


class CatDogDataset(Dataset):
    def __init__(self, img_dir, ann_dir, input_img_size: int, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.input_img_size = input_img_size
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # Default to -1 if unknown label
            label = self.label_map.get(name, -1)  
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        width, height, objects = self.parse_annotation(ann_path)

        scaler_x = width / self.input_img_size
        scaler_y = height / self.input_img_size

        bboxes = []
        for obj in objects:
            xmin = obj['bbox'][0] / scaler_x
            ymin = obj['bbox'][1] / scaler_y
            xmax = obj['bbox'][2] / scaler_x
            ymax = obj['bbox'][3] / scaler_y
            # in your assignment 4, you need to convert bbox into 
            # [x, y, w, h] and value range [0, 1]
            bboxes.append([xmin, ymin, xmax, ymax])  

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(
            [obj["label"] for obj in objects], 
            dtype=torch.int64
        )

        if self.transform:
            image = self.transform(image)

        return image, bboxes, labels
