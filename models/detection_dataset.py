import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Dict, List, Tuple
import logging
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
import torch.optim as optim
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from datetime import datetime

class EnhancedObjectDetectionDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 annotation_file: str, 
                 split: str,
                 split_ratio: float = 0.8,
                 image_size: int = 416,
                 num_classes: int = 10,
                 seed: int = 42):
        """
        Args:
            image_dir: Path to directory containing all images
            annotation_file: Path to COCO annotation JSON file
            split: Dataset split ('train' or 'val')
            split_ratio: Ratio of data to use for training
            image_size: Target image size for resizing
            num_classes: Number of object classes
            seed: Random seed for reproducible splits
        """
        self.image_dir = Path(image_dir)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Setup logging
        self.logger = logging.getLogger(f"{split}_dataset")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initializing {split} dataset...")
        self.logger.info(f"Image directory: {self.image_dir}")
        self.logger.info(f"Annotation file: {annotation_file}")
        
        # Load COCO annotations
        try:
            with open(annotation_file, 'r') as f:
                self.coco = json.load(f)
            self.logger.info("Successfully loaded COCO annotations")
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            raise
            
        # Get all image ids and create split
        all_image_ids = [img['id'] for img in self.coco['images']]
        
        # Create reproducible random split
        np.random.seed(seed)
        np.random.shuffle(all_image_ids)
        split_idx = int(len(all_image_ids) * split_ratio)
        
        # Select image ids for this split
        if split == 'train':
            self.image_ids = all_image_ids[:split_idx]
        else:  # 'val'
            self.image_ids = all_image_ids[split_idx:]
            
        self.logger.info(f"Using {len(self.image_ids)} images for {split}")
        
        # Create image id to annotations mapping
        self.annotations = {}
        
        # Create category id to name mapping
        self.cat_id_to_name = {
            cat['id']: cat['name'] 
            for cat in self.coco['categories']
        }
        
        # Only keep images for this split
        for img in self.coco['images']:
            if img['id'] in self.image_ids:
                self.annotations[img['id']] = {
                    'file_name': img['file_name'].replace("data/", ""),
                    'width': img['width'],
                    'height': img['height'],
                    'objects': []
                }
            
        # Only keep annotations for images in this split
        ann_count = 0
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id in self.annotations:
                # Verify bbox format
                bbox = ann['bbox']
                if len(bbox) != 4:
                    self.logger.warning(f"Skipping invalid bbox: {bbox}")
                    continue
                    
                self.annotations[img_id]['objects'].append({
                    'bbox': bbox,  # [x, y, width, height]
                    'category_id': ann['category_id']
                })
                ann_count += 1
                
        self.logger.info(f"Loaded {len(self.annotations)} images and {ann_count} annotations for {split}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create flow directory path
        self.flow_dir = self.image_dir.parent / 'flow'
        if not self.flow_dir.exists():
            self.logger.warning(f"Flow directory not found: {self.flow_dir}")
            
        # Verify image files exist
        self._verify_images()
        
    def _verify_images(self):
        """Verify all image files exist"""
        missing_images = []
        for img_id in self.image_ids:
            img_path = self.image_dir / self.annotations[img_id]['file_name'].replace("data/", "")
            if not img_path.exists():
                missing_images.append(str(img_path))
        
        if missing_images:
            self.logger.warning(f"Missing {len(missing_images)} images")
            self.logger.debug(f"First few missing images: {missing_images[:5]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def convert_bbox(self, bbox: List[float], orig_w: int, orig_h: int) -> torch.Tensor:
        """
        Convert COCO bbox [x, y, w, h] to normalized [x_center, y_center, w, h]
        """
        x, y, w, h = bbox
        
        # Convert to center coordinates
        x_center = x + w/2
        y_center = y + h/2
        
        # Normalize coordinates
        x_center /= orig_w
        y_center /= orig_h
        w /= orig_w
        h /= orig_h
        
        return torch.tensor([x_center, y_center, w, h], dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        img_info = self.annotations[img_id]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
        
        # Original image dimensions
        orig_h, orig_w = img.shape[:2]
        
        # Load optical flow
        flow_path = self.flow_dir / f"{img_path.stem}_flow.npy"
        if flow_path.exists():
            try:
                flow = np.load(str(flow_path))
            except Exception as e:
                self.logger.warning(f"Error loading flow {flow_path}: {str(e)}")
                flow = np.zeros((orig_h, orig_w, 2), dtype=np.float32)
        else:
            flow = np.zeros((orig_h, orig_w, 2), dtype=np.float32)
        
        # Resize image and flow
        img = cv2.resize(img, (self.image_size, self.image_size))
        flow = cv2.resize(flow, (self.image_size, self.image_size))
        
        # Prepare labels
        labels = []
        for obj in img_info['objects']:
            category_id = obj['category_id']
            bbox = self.convert_bbox(obj['bbox'], orig_w, orig_h)
            labels.append(torch.cat([torch.tensor([category_id]), bbox]))
        
        if not labels:
            labels = torch.zeros((1, 5))  # One dummy box with class 0
        else:
            labels = torch.stack(labels)
        
        # Convert image to tensor and normalize
        img = self.transform(img)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        return {
            'image': img,
            'flow': flow,
            'labels': labels,
            'img_path': str(img_path)
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader
    """
    images = torch.stack([item['image'] for item in batch])
    flows = torch.stack([item['flow'] for item in batch])
    
    # Find max number of labels in batch
    max_labels = max(item['labels'].shape[0] for item in batch)
    
    # Pad labels to same length
    padded_labels = []
    for item in batch:
        if item['labels'].shape[0] < max_labels:
            padding = torch.zeros((max_labels - item['labels'].shape[0], 5))
            labels = torch.cat([item['labels'], padding], dim=0)
        else:
            labels = item['labels']
        padded_labels.append(labels)
    
    labels = torch.stack(padded_labels)
    
    return {
        'image': images,
        'flow': flows,
        'labels': labels,
        'img_path': [item['img_path'] for item in batch]
    }
