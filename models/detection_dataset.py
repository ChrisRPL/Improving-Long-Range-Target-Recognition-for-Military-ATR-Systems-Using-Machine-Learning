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
        """Load labels from txt file."""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        try:
            if not label_path.exists():
                print(f"No label file found for {img_path}")
                return torch.zeros((1, 5))

            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 9:  # class_id + 8 coordinates
                        class_id = int(float(values[0]))  # Class ID should be an integer
                        if not (0 <= class_id < self.num_classes):
                            print(f"Invalid class ID {class_id} in {label_path}")
                            continue
                        
                        # Convert coordinates to x, y, w, h format
                        x_coords = [float(values[i]) for i in [1,3,5,7]]
                        y_coords = [float(values[i]) for i in [2,4,6,8]]
                    
                        x1, x2 = min(x_coords), max(x_coords)
                        y1, y2 = min(y_coords), max(y_coords)
                    
                        # Calculate center coordinates and dimensions
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                    
                        labels.append([class_id, x_center, y_center, width, height])

            if not labels:
                print(f"No valid labels found in {label_path}")
                return torch.zeros((1, 5))

            label_tensor = torch.tensor(labels, dtype=torch.float32)
            print(f"Loaded {len(labels)} labels from {label_path}")
            return label_tensor

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
