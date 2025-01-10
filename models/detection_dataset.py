import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
from torchvision import transforms

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
            
        labels = np.loadtxt(str(label_path))
        labels = torch.from_numpy(labels).float()
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)
        return labels
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load optical flow
        flow_path = self.flow_dir / f"{img_path.stem}_flow.npy"
        flow = np.load(str(flow_path))
        
        # Load labels
        labels = self.load_labels(img_path)
        
        # Apply transforms
        img = self.transforms(img)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        return {
            'image': img,
            'flow': flow,
            'labels': labels,  # [class_id, x, y, w, h]
            'img_path': str(img_path)
        }
