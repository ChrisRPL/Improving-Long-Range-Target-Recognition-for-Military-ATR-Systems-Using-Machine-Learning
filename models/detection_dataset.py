import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
from torchvision import transforms
from typing import Dict, List

class EnhancedObjectDetectionDataset(Dataset):
    def __init__(self, data_root: Path, split: str, image_size=416):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Setup paths
        self.images_dir = self.data_root / split / 'images'
        self.labels_dir = self.data_root / split / 'labels'
        self.flow_dir = self.data_root / split / 'flow'
        
        # Get all image paths
        self.image_paths = sorted(list(self.images_dir.glob('*.jpg')))
        print(f"Found {len(self.image_paths)} images in {self.images_dir}")
        
        # Verify label files exist
        self.has_labels = []
        for img_path in self.image_paths:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            self.has_labels.append(label_path.exists())
        
        print(f"Found {sum(self.has_labels)} images with labels")

    def load_labels(self, img_path):
        """Load YOLO format labels."""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            return torch.zeros((1, 5))  # Return dummy label instead of empty tensor
            
        try:
            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 5:
                        labels.append([float(x) for x in values])
            
            if not labels:
                return torch.zeros((1, 5))  # Return dummy label if file is empty
                
            return torch.tensor(labels)
        except Exception as e:
            print(f"Error loading labels from {label_path}: {str(e)}")
            return torch.zeros((1, 5))  # Return dummy label on error

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and process image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Load optical flow
        flow_path = self.flow_dir / f"{img_path.stem}_flow.npy"
        if flow_path.exists():
            flow = np.load(str(flow_path))
        else:
            flow = np.zeros((self.image_size, self.image_size, 2), dtype=np.float32)
        
        # Load labels
        labels = self.load_labels(img_path)
        
        # Convert to tensors
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        flow = torch.from_numpy(flow).float().permute(2, 0, 1)
        
        return {
            'image': img,
            'flow': flow,
            'labels': labels,
            'img_path': str(img_path)
        }

def collate_fn(batch):
    """Custom collate function to handle variable number of labels."""
    images = torch.stack([item['image'] for item in batch])
    flows = torch.stack([item['flow'] for item in batch])
    
    # Find max number of labels in batch
    max_labels = max(item['labels'].shape[0] for item in batch)
    
    # Pad labels to max length
    padded_labels = []
    for item in batch:
        num_labels = item['labels'].shape[0]
        if num_labels == 0:
            # Handle empty labels
            padding = torch.full((max_labels, 5), -1)
            padded_labels.append(padding)
        else:
            padding = torch.full((max_labels - num_labels, 5), -1)
            padded = torch.cat([item['labels'], padding], dim=0)
            padded_labels.append(padded)
    
    labels = torch.stack(padded_labels)
    
    return {
        'image': images,
        'flow': flows,
        'labels': labels,
    }
