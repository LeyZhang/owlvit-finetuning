import json
import os
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
from src.matcher import BoxUtils
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from urllib.request import urlretrieve
import torchvision
import os
from torchvision.ops import box_convert as _box_convert
from PIL import Image, ImageDraw

class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file):
        self.ann_file = ann_file
        super(CocoDataset, self).__init__(root = img_folder, annFile = ann_file)
        with open(self.ann_file) as f:
            categories = json.load(f)["categories"]
        self.id2label = {category["id"]: category["name"] for category in categories}
        self.label2id = {v:k for k, v in self.id2label.items()}
    
    @staticmethod          
    def _box_xywh_to_cxcywh(
        boxes_batch: torch.tensor,  # [N, 4]
    ):
        return _box_convert(boxes_batch, 'xywh', 'cxcywh')

    @staticmethod          
    def _box_cxcywh_to_xyxy(
        boxes_batch: torch.tensor,  # [N, 4]
    ):
        return _box_convert(boxes_batch, 'cxcywh', 'xyxy')

    def _rescale_bboxes(self,out_bbox: torch.tensor, size: torch.tensor):
        img_h, img_w  = size.unbind(-1)
        b = self._box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def formatted_anns(self, labels):
        annotations = []
        bboxes = labels['boxes']
        for i in range(0, len(labels['class_labels'])):
            new_ann = {
                "image_id": labels['image_id'],
                "category_id": labels['class_labels'][i],
                "isCrowd": 0,
                "area": labels['area'][i],
                "bbox": self._rescale_bboxes(bboxes[i], labels['orig_size']),
            }
            annotations.append(new_ann)
        return annotations

    def draw_annotation(self, idx: int, path:str):
        img, cocoinfo = super(CocoDataset, self).__getitem__(idx)
        outputs = self.__getitem__(idx)
        annotations = self.formatted_anns(outputs['labels'])
        draw = ImageDraw.Draw(img, "RGBA")
        for annotation in annotations:
            box = annotation['bbox']
            class_idx = annotation['category_id']
            x1, y1, x2, y2 = box.unbind(-1)
            draw.rectangle((x1, y1, x2, y2), outline='red', width=1)
            draw.text((x1, y2), self.id2label[int(class_idx)], fill='white')
        img.save(f"{path}/{idx}.jpg")

    
class OwlDataset(CocoDataset):
    def __init__(self, processor, img_folder, ann_file, train=True):
        self.ann_file = ann_file
        super(OwlDataset, self).__init__(img_folder=img_folder, ann_file= ann_file)
        self.processor = processor
        with open(self.ann_file) as f:
            categories = json.load(f)["categories"]
        self.id2label = {category["id"]: category["name"] for category in categories}
        self.label2id = {v:k for k, v in self.id2label.items()}

    def _getinfo(self, cocoinfo, img):
        texts = ['a photo of ' + self.id2label[info['category_id']] for info in cocoinfo]
        #  dedupe
        texts = list(set(texts))
        h, w = img.size
        bboxes = []
        for info in cocoinfo:
            bbox = torch.Tensor(info['bbox'])*torch.tensor([1/w, 1/h, 1/w, 1/h], dtype=torch.float32)
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes,0)
        class_labels = [info['category_id']-1 for info in cocoinfo]
        areas = [info['area'] for info in cocoinfo]
        return texts, bboxes , class_labels, areas
 
    def __getitem__(self, idx:int):
        img, cocoinfo = super(OwlDataset, self).__getitem__(idx)
        texts, bboxes, class_labels, areas = self._getinfo(cocoinfo, img)
        image_id = self.ids[idx]
        h, w = img.size
        
        inputs = self.processor(images=img, text = texts, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0) 
        labels =  {
            "image_id": torch.Tensor(image_id).int(), 
            "class_labels": torch.Tensor(class_labels).int(), 
            "area": torch.Tensor(areas),
            "boxes": self._box_xywh_to_cxcywh(bboxes),
            "orig_size":torch.Tensor([h, w])
        }
                   
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "labels": labels
        }


class LvisDataSet(CocoDataset):
    def __init__(self, processor, img_folder, ann_file, train=True):
        super(LvisDataSet, self).__init__(img_folder=img_folder, ann_file= ann_file)
        self.processor = processor
        pass

    def _load_image(self, id: int) -> Image.Image:
        url = self.coco.loadImgs(id)[0]["coco_url"]
        file_name = os.path.join(self.root, url.split("/")[-1])
        if not os.path.exists(file_name):
            urlretrieve(url, file_name)
        return Image.open(file_name).convert("RGB")

    def _getinfo(self, cocoinfo, img):
        texts = ['a photo of ' + self.id2label[info['category_id']] for info in cocoinfo]
        #  dedupe
        texts = list(set(texts))
        h, w = img.size
        bboxes = []
        for info in cocoinfo:
            bbox = torch.Tensor(info['bbox'])*torch.tensor([1/w, 1/h, 1/w, 1/h], dtype=torch.float32)
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes,0)
        class_labels = [info['category_id'] for info in cocoinfo]
        areas = [info['area'] for info in cocoinfo]
        return texts, bboxes , class_labels, areas

    def __getitem__(self, idx:int):
        img, cocoinfo = super(LvisDataSet, self).__getitem__(idx)
        texts, bboxes, class_labels, areas = self._getinfo(cocoinfo, img)
        image_id = self.ids[idx]
        h, w = img.size
        
        inputs = self.processor(images=img, text = texts, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0) 
        labels =  {
            "image_id": torch.Tensor(image_id).int(), 
            "class_labels": torch.Tensor(class_labels).int(), 
            "area": torch.Tensor(areas),
            "boxes": self._box_xywh_to_cxcywh(bboxes),
            "orig_size":torch.Tensor([h, w])
        }
                   
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "labels": labels
        }



class OwlDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, device, shuffle):
        super(OwlDataLoader, self).__init__(
            dataset, 
            batch_size=batch_size , 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn = self.collate_fn
        )
        self.device = device

    
    def collate_fn(self, batch):
        input_ids = torch.cat([item["input_ids"] for item in batch],0).int().to(self.device)
        attention_mask = torch.cat([item["attention_mask"] for item in batch], 0).int().to(self.device)
        pixel_values = torch.cat([item["pixel_values"] for item in batch], 0).to(self.device)
        labels = []
        for item in batch:
            for (key, value) in item["labels"].items():
                item["labels"][key] = torch.Tensor(value).to(self.device)
            labels.append(item["labels"])
        batch = {}
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["pixel_values"] = pixel_values
        batch["labels"] = labels
        return batch






def get_owl_dataloaders(cfg, processor, device):
    train_dataset, test_dataset = get_datasets(cfg, processor)
    print("Number of training examples:", len(train_dataset))
    print("Number of testing examples:", len(test_dataset))
    
    train_dataloader = OwlDataLoader(train_dataset,batch_size=cfg['batch_size'], shuffle=True, num_workers=1, device=device)
    test_dataloader = OwlDataLoader(test_dataset,batch_size=cfg['batch_size'], shuffle=True, num_workers=1, device=device) 
    return train_dataloader, test_dataloader

def get_owl_datasets(cfg, processor):
    train_dataset = OwlDataset(
                        processor, 
                        img_folder=cfg['train_images_path'], 
                        ann_file=cfg['train_annotations_path'])
    
    test_dataset = OwlDataset(
                        processor, 
                        img_folder=cfg['train_images_path'],
                        ann_file=cfg['test_annotations_path'], 
                        train=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of testing examples:", len(test_dataset))
    return train_dataset, test_dataset