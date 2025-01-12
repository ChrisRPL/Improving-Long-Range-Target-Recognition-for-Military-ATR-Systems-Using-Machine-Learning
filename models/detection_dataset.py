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
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def load_labels(self, img_path):
        """
        Load labels from txt file.
        Format: class_id x1 y1 x2 y1 x1 y2 x2 y2
        """
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        try:
            if not label_path.exists():
                return torch.zeros((1, 5))

            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 9:  # class_id + 8 coordinates
                        # Parse values
                        class_id = int(values[0])
                        coords = [float(v) for v in values[1:]]
                    
                        # Convert from x1,y1,x2,y1,x1,y2,x2,y2 to YOLO format (x_center, y_center, width, height)
                        x1 = min(coords[0], coords[2], coords[4], coords[6])
                        x2 = max(coords[0], coords[2], coords[4], coords[6])
                        y1 = min(coords[1], coords[3], coords[5], coords[7])
                        y2 = max(coords[1], coords[3], coords[5], coords[7])
                    
                        # Calculate center coordinates and dimensions
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                    
                        # Append in YOLO format: [class_id, x_center, y_center, width, height]
                        labels.append([class_id, x_center, y_center, width, height])

            if not labels:
                return torch.zeros((1, 5))

            return torch.tensor(labels, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading labels from {label_path}: {str(e)}")
            return torch.zeros((1, 5))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
    
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Load optical flow
        flow_path = self.flow_dir / f"{img_path.stem}_flow.npy"
        if flow_path.exists():
            flow = np.load(str(flow_path))
        else:
            flow = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
    
        # Load and verify labels
        labels = self.load_labels(img_path)
    
        # For debugging (first few batches)
        if idx < 5:
            print(f"\nSample {idx}:")
            print(f"Image path: {img_path}")
            print(f"Label path: {self.labels_dir / f'{img_path.stem}.txt'}")
            print(f"Labels loaded: {labels}")
    
        # Resize both image and flow
        img = cv2.resize(img, (self.image_size, self.image_size))
        flow = cv2.resize(flow, (self.image_size, self.image_size))
    
        # Convert to tensors and normalize
        img = self.transform(img)
        flow = torch.from_numpy(flow).float().permute(2, 0, 1)
    
        return {
            'image': img,
            'flow': flow,
            'labels': labels,
            'img_path': str(img_path)
        }
