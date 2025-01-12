import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from typing import Dict, List, Tuple
import random

class EnhancedObjectDetectionDataset(Dataset):
    def __init__(self, 
                 image_dir: str,
                 annotation_file: str, 
                 split: str,
                 split_ratio: float = 0.8,
                 image_size: int = 416,
                 num_classes: int = 10,
                 augment: bool = True,
                 seed: int = 42):
        """
        Enhanced dataset class with proper augmentations and error handling
        """
        self.image_dir = Path(image_dir)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment and split == 'train'
        
        # Setup logging
        self.logger = logging.getLogger(f"{split}_dataset")
        self.logger.setLevel(logging.INFO)
        
        # Load annotations
        try:
            with open(annotation_file, 'r') as f:
                self.coco = json.load(f)
            self.logger.info(f"Loaded {len(self.coco['images'])} images from annotations")
        except Exception as e:
            self.logger.error(f"Error loading annotations: {str(e)}")
            raise
        
        # Create reproducible split
        random.seed(seed)
        all_image_ids = [img['id'] for img in self.coco['images']]
        random.shuffle(all_image_ids)
        split_idx = int(len(all_image_ids) * split_ratio)
        
        self.image_ids = all_image_ids[:split_idx] if split == 'train' else all_image_ids[split_idx:]
        
        # Create image id to annotations mapping
        self.annotations = {}
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco['categories']}
        
        # Process annotations
        for img in self.coco['images']:
            if img['id'] in self.image_ids:
                self.annotations[img['id']] = {
                    'file_name': img['file_name'].replace("data/", ""),
                    'width': img['width'],
                    'height': img['height'],
                    'objects': []
                }
        
        # Process object annotations
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id in self.annotations:
                bbox = ann['bbox']
                if len(bbox) != 4:
                    self.logger.warning(f"Skipping invalid bbox: {bbox}")
                    continue
                
                self.annotations[img_id]['objects'].append({
                    'bbox': bbox,
                    'category_id': ann['category_id']
                })
        
        # Setup augmentations
        self.transform = self._get_transforms()
        
        # Verify all files exist
        self._verify_files()
        
    def _verify_files(self):
        """Verify all required files exist"""
        missing_images = []
        missing_flows = []
        
        for img_id in self.image_ids:
            img_path = self.image_dir / self.annotations[img_id]['file_name'].replace("data/", "")
            flow_path = self.image_dir.parent / 'flow' / f"{img_path.stem}_flow.npy"
            
            if not img_path.exists():
                missing_images.append(str(img_path))
            if not flow_path.exists():
                missing_flows.append(str(flow_path))
        
        if missing_images:
            self.logger.warning(f"Missing {len(missing_images)} images")
        if missing_flows:
            self.logger.warning(f"Missing {len(missing_flows)} flow files")
    
    def _get_transforms(self):
        """Get augmentation pipeline"""
        if self.augment:
            return A.Compose([
            A.RandomScale(scale_limit=0.2),
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.HueSaturationValue()
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(),
                A.MotionBlur()
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.3
        ))
        else:
            return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids']
        ))
    
    def __len__(self):
        return len(self.image_ids)
    
    def _load_flow(self, img_path: Path) -> np.ndarray:
        """Load and process optical flow"""
        flow_path = self.image_dir.parent / 'flow' / f"{img_path.stem}_flow.npy"
        try:
            flow = np.load(str(flow_path))
            # Resize flow to match image_size
            flow = cv2.resize(flow, (self.image_size, self.image_size))
            return flow.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"Error loading flow {flow_path}: {str(e)}")
            return np.zeros((self.image_size, self.image_size, 2), dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        img_info = self.annotations[img_id]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to target size
            image = cv2.resize(image, (self.image_size, self.image_size))
        except Exception as e:
            self.logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
        
        # Load flow
        flow = self._load_flow(img_path)
        
        # Prepare boxes and labels
        boxes = []
        labels = []
        for obj in img_info['objects']:
            # Scale boxes according to resize ratio
            orig_h, orig_w = img_info['height'], img_info['width']
            x, y, w, h = obj['bbox']
            
            # Convert to relative coordinates
            x = x / orig_w
            y = y / orig_h
            w = w / orig_w
            h = h / orig_h
            
            boxes.append([x, y, w, h])
            labels.append(obj['category_id'])
        
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        transformed = self.transform(
            image=image,
            bboxes=boxes,
            category_ids=labels
        )
        
        image = transformed['image']
        
        # Handle transformed boxes safely
        transformed_bboxes = transformed.get('bboxes', [])
        transformed_category_ids = transformed.get('category_ids', [])
        
        # Convert to numpy arrays with proper handling of empty case
        if len(transformed_bboxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int64)
        else:
            boxes = np.array(transformed_bboxes, dtype=np.float32)
            labels = np.array(transformed_category_ids, dtype=np.int64)
        
        # Convert flow to tensor and ensure correct shape
        flow = torch.from_numpy(flow.transpose(2, 0, 1))
        
        # Convert boxes to cxcywh format
        if boxes.shape[0] > 0:
            boxes = self._convert_to_cxcywh(boxes)
        else:
            boxes = torch.zeros((0, 4))
        
        return {
            'image': image,
            'flow': flow,
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'img_id': img_id,
            'img_path': str(img_path)
        }
    
    @staticmethod
    def _convert_to_cxcywh(boxes):
        """Convert COCO format (x,y,w,h) to cxcywh format"""
        boxes = torch.as_tensor(boxes)
        x, y, w, h = boxes.unbind(-1)
        cx = x + w/2
        cy = y + h/2
        return torch.stack([cx, cy, w, h], dim=-1)

def collate_fn(batch):
    """Custom collate function for dataloader"""
    # All images and flows should already be the same size due to resize in __getitem__
    images = torch.stack([item['image'] for item in batch])
    flows = torch.stack([item['flow'] for item in batch])
    
    # Pad boxes and labels to same length
    max_boxes = max(len(item['boxes']) for item in batch)
    
    if max_boxes == 0:
        boxes = torch.zeros(len(batch), 0, 4)
        labels = torch.zeros(len(batch), 0, dtype=torch.int64)
    else:
        boxes = torch.zeros(len(batch), max_boxes, 4)
        labels = torch.zeros(len(batch), max_boxes, dtype=torch.int64)
        
        for idx, item in enumerate(batch):
            if len(item['boxes']) > 0:
                boxes[idx, :len(item['boxes'])] = item['boxes']
                labels[idx, :len(item['labels'])] = item['labels']
    
    # Create batch
    batch_dict = {
        'image': images,
        'flow': flows,
        'boxes': boxes,
        'labels': labels,
        'img_ids': [item['img_id'] for item in batch],
        'img_paths': [item['img_path'] for item in batch]
    }
    
    return batch_dict
    
class DataModule:
    """Data module to handle all data-related operations"""
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 416,
        split_ratio: float = 0.8,
        augment: bool = True,
        seed: int = 42
    ):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        
        # Adjust number of workers based on system CPU count
        cpu_count = os.cpu_count()
        suggested_workers = max(1, (cpu_count or 2) - 1)  # Leave one CPU free
        self.num_workers = min(num_workers, suggested_workers)
        
        self.image_size = image_size
        self.split_ratio = split_ratio
        self.augment = augment
        self.seed = seed
        
        # Validate paths
        self.image_dir = self.dataset_dir / 'data'
        self.annotation_file = self.dataset_dir / 'coco.json'
        self.flow_dir = self.dataset_dir / 'flow'
        
        if not self.image_dir.exists():
            raise ValueError(f"Images directory not found: {self.image_dir}")
        if not self.annotation_file.exists():
            raise ValueError(f"Annotation file not found: {self.annotation_file}")
        if not self.flow_dir.exists():
            raise ValueError(f"Flow directory not found: {self.flow_dir}")
        
        # Load category information
        with open(self.annotation_file) as f:
            coco_data = json.load(f)
            self.categories = coco_data['categories']
            self.num_classes = len(self.categories)
    
    def setup(self):
        """Setup train and validation datasets"""
        self.train_dataset = EnhancedObjectDetectionDataset(
            image_dir=self.image_dir,
            annotation_file=self.annotation_file,
            split='train',
            split_ratio=self.split_ratio,
            image_size=self.image_size,
            num_classes=self.num_classes,
            augment=self.augment,
            seed=self.seed
        )
        
        self.val_dataset = EnhancedObjectDetectionDataset(
            image_dir=self.image_dir,
            annotation_file=self.annotation_file,
            split='val',
            split_ratio=self.split_ratio,
            image_size=self.image_size,
            num_classes=self.num_classes,
            augment=False,
            seed=self.seed
        )
    
    def train_dataloader(self):
        """Create training dataloader"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def get_category_names(self):
        """Get mapping of category IDs to names"""
        return {cat['id']: cat['name'] for cat in self.categories}
