import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.detection_model import create_model
from utils.annotations import load_yolo_annotations
import os
import cv2
import numpy as np
from torchvision.transforms import functional as F
import argparse

class YoloDataset(Dataset):
    def __init__(self, image_paths, flow_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.flow_paths = flow_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        flow = np.load(self.flow_paths[idx])
        bboxes = self.annotations[idx]['boxes']
        labels = self.annotations[idx]['labels']

        if self.transform:
            augmented = self.transform(image=image, masks=[flow])
            image = augmented['image']
            flow = augmented['masks'][0]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {'boxes': bboxes, 'labels': labels}

        return image, flow, target

def yolo_loss(bboxes_pred, class_logits_pred, bboxes_true, class_labels_true):
    # Calculate IoU
    iou = box_iou(bboxes_pred, bboxes_true)

    # Loss functions
    loss_bbox = 1 - iou.mean()
    loss_class = F.cross_entropy(class_logits_pred, class_labels_true)

    return loss_bbox + loss_class

def train_model(model, dataloader, optimizer, device, num_epochs=25):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, flows, targets in dataloader:
            images = images.to(device)
            flows = flows.to(device)
            bboxes_true = targets['boxes'].to(device)
            class_labels_true = targets['labels'].to(device)

            optimizer.zero_grad()
            bboxes_pred, class_logits_pred = model(images, flows)

            # Compute loss
            loss = yolo_loss(bboxes_pred, class_logits_pred, bboxes_true, class_labels_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs} Loss: {epoch_loss:.4f}')

    print('Training complete')
    return model

def load_dataset(dataset_type):
    if dataset_type == 'mwir':
        image_dir = 'data/mwir_videos'
    elif dataset_type == 'visible':
        image_dir = 'data/visible_videos'
    elif dataset_type == 'fused':
        image_dir = 'data/fused_videos'
    elif dataset_type == 'dsgan':
        image_dir = 'data/dsgan_videos'
    else:
        raise ValueError("Invalid dataset type")

    annotation_dir = 'data/annotations'
    image_paths = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir))]
    flow_paths = [os.path.join(image_dir, f.replace('.jpg', '.npy')) for f in sorted(os.listdir(image_dir))]
    annotations = [load_yolo_annotations(os.path.join(annotation_dir, ann)) for ann in sorted(os.listdir(annotation_dir))]
    
    dataset = YoloDataset(image_paths, flow_paths, annotations)
    return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Detection Model')
    parser.add_argument('--dataset_type', type=str, required=True, help='Type of dataset: mwir, visible, fused, dsgan')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes for the model')
    args = parser.parse_args()

    dataloader = load_dataset(args.dataset_type)
    
    model = create_model(num_classes=args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trained_model = train_model(model, dataloader, optimizer, device, num_epochs=25)
    
    # Save trained model
    torch.save(trained_model.state_dict(), f'outputs/enhanced_detection_model_{args.dataset_type}.pth')

