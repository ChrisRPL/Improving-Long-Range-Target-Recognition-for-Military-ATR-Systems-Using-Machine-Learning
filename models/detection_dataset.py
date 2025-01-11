import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
from torchvision import transforms
from typing import Dict, List

class EnhancedObjectDetectionDataset(Dataset):
    def __init__(self, data_root: Path, split: str, image_size=640):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Setup paths
        self.images_dir = self.data_root / split / 'images'
        self.labels_dir = self.data_root / split / 'labels'
        self.flow_dir = self.data_root / split / 'flow'
        
        # Get all image paths
        self.image_paths = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def load_labels(self, img_path):
        """Load YOLO format labels."""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            return torch.zeros((0, 5))  # Return empty tensor if no labels
            
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) == 5:  # class, x, y, w, h
                    labels.append([float(x) for x in values])
        
        return torch.tensor(labels) if labels else torch.zeros((0, 5))
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load optical flow
        flow_path = self.flow_dir / f"{img_path.stem}_flow.npy"
        if flow_path.exists():
            flow = np.load(str(flow_path))
        else:
            flow = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
        
        # Resize image and flow to target size
        img = cv2.resize(img, (self.image_size, self.image_size))
        flow = cv2.resize(flow, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        img = self.transforms(img)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        # Load and process labels
        labels = self.load_labels(img_path)
        
        return {
            'image': img,
            'flow': flow,
            'labels': labels,
            'img_path': str(img_path)
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable number of labels."""
    images = torch.stack([item['image'] for item in batch])
    flows = torch.stack([item['flow'] for item in batch])
    
    # Pad labels to same length
    max_labels = max(item['labels'].shape[0] for item in batch)
    labels = []
    for item in batch:
        if item['labels'].shape[0] == 0:
            padded = torch.zeros((max_labels, 5))
        else:
            padded = torch.zeros((max_labels, 5))
            padded[:item['labels'].shape[0]] = item['labels']
        labels.append(padded)
    
    labels = torch.stack(labels)
    paths = [item['img_path'] for item in batch]
    
    return {
        'image': images,
        'flow': flows,
        'labels': labels,
        'img_path': paths
    }
