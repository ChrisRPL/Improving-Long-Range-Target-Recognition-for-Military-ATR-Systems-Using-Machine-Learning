import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from torchvision import transforms
import os
import warnings
from typing import Dict, List
from models.detection_dataset import EnhancedObjectDetectionDataset
from models.detection_model import EnhancedDetectionModel
warnings.filterwarnings('ignore')
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import torch.optim as optim


sys.path.append(str(Path(__file__).parent.parent))

def setup_logger(output_dir):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    log_file = Path(output_dir) / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def train_model(model, train_loader, val_loader, config, output_dir):
    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4))
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get('lr', 1e-4),
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    # Initialize metrics
    best_map50 = 0
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'mAP50': [], 'mAP50-95': [],
        'precision': [], 'recall': [],
        'per_class_map': []
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['epochs']):
        logger.info(f"\nStarting epoch {epoch+1}/{config['epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0
        model.map_metric.reset()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            images = batch['image'].to(device)
            flows = batch['flow'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.cuda.amp.autocast():
                pred_boxes, pred_logits = model(images, flows)
                loss = model.loss_fn(pred_boxes, pred_logits, labels[..., 1:], labels[..., 0])
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Format predictions and targets for metrics
            predictions = []
            targets = []
            
            for i in range(len(labels)):
                valid_mask = labels[i, :, 0] >= 0
                targets.append({
                    'boxes': labels[i, valid_mask, 1:].cpu(),
                    'labels': labels[i, valid_mask, 0].long().cpu()
                })
                
                scores = torch.softmax(pred_logits[i], dim=-1)
                max_scores, pred_labels = scores.max(dim=-1)
                
                # Filter predictions by confidence
                conf_mask = max_scores > 0.05
                predictions.append({
                    'boxes': pred_boxes[i][conf_mask].cpu(),
                    'scores': max_scores[conf_mask].cpu(),
                    'labels': pred_labels[conf_mask].cpu()
                })
            
            model.map_metric.update(predictions, targets)
        
        # Calculate training metrics
        train_metrics = model.map_metric.compute()
        avg_train_loss = train_loss / len(train_loader)
        model.map_metric.reset()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(device)
                flows = batch['flow'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.cuda.amp.autocast():
                    pred_boxes, pred_logits = model(images, flows)
                    loss = model.loss_fn(pred_boxes, pred_logits, labels[..., 1:], labels[..., 0])
                
                val_loss += loss.item()
                
                # Update metrics
                predictions = []
                targets = []
                
                for i in range(len(labels)):
                    valid_mask = labels[i, :, 0] >= 0
                    targets.append({
                        'boxes': labels[i, valid_mask, 1:].cpu(),
                        'labels': labels[i, valid_mask, 0].long().cpu()
                    })
                    
                    scores = torch.softmax(pred_logits[i], dim=-1)
                    max_scores, pred_labels = scores.max(dim=-1)
                    
                    conf_mask = max_scores > 0.05
                    predictions.append({
                        'boxes': pred_boxes[i][conf_mask].cpu(),
                        'scores': max_scores[conf_mask].cpu(),
                        'labels': pred_labels[conf_mask].cpu()
                    })
                
                model.map_metric.update(predictions, targets)
        
        # Calculate validation metrics
        val_metrics = model.map_metric.compute()
        avg_val_loss = val_loss / len(val_loader)
        model.map_metric.reset()
        
        # Update metrics history
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['mAP50'].append(val_metrics['map_50'].item())
        metrics_history['mAP50-95'].append(val_metrics['map'].item())
        metrics_history['precision'].append(val_metrics['map_per_class'].mean().item())
        metrics_history['recall'].append(val_metrics['mar_100'].item())
        metrics_history['per_class_map'].append(val_metrics['map_per_class'].tolist())
        
        # Log metrics
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"mAP50: {val_metrics['map_50']:.4f}")
        logger.info(f"mAP50-95: {val_metrics['map']:.4f}")
        logger.info(f"Precision: {val_metrics['map_per_class'].mean():.4f}")
        logger.info(f"Recall: {val_metrics['mar_100']:.4f}")
        
        # Save best model
        if val_metrics['map_50'] > best_map50:
            best_map50 = val_metrics['map_50']
            save_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, str(save_path))
            logger.info(f"Saved new best model with mAP50: {best_map50:.4f}")
        
        # Save metrics history
        torch.save(metrics_history, output_dir / 'metrics_history.pt')
        
        # Plot metrics
        plot_metrics(metrics_history, output_dir / 'metrics.png', config)
    
    return model, metrics_history

def plot_metrics(metrics_history, save_path, config):
    """Plot training metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train')
    plt.plot(metrics_history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot mAP
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['mAP50'], label='mAP50')
    plt.plot(metrics_history['mAP50-95'], label='mAP50-95')
    plt.title('Mean Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['precision'])
    plt.title('Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    
    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['recall'])
    plt.title('Average Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Detection Model')
    pparser = argparse.ArgumentParser(description='Train Enhanced Detection Model')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data.yaml file')
    parser.add_argument('--dataset-dir', type=str, required=True,
                      help='Path to dataset root directory (containing data/, flow/, coco.json)')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                      help='Train/val split ratio')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--img-size', type=int, default=416,
                      help='Input image size')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='runs/train',
                      help='Output directory')
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Load config
    config = yaml.safe_load(open(args.data))
    
    # Update config
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'img_size': args.img_size,
    })
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / 'data'
    coco_file = dataset_dir / 'coco.json'
    
    # Verify paths
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not coco_file.exists():
        raise ValueError(f"COCO annotations file not found: {coco_file}")
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Images directory: {images_dir}")
    print(f"COCO annotations: {coco_file}")
    
    # Create datasets
    train_dataset = EnhancedObjectDetectionDataset(
        image_dir=images_dir,
        annotation_file=coco_file,
        split='train',
        split_ratio=args.split_ratio,
        image_size=args.img_size,
        num_classes=config['nc']
    )
    
    val_dataset = EnhancedObjectDetectionDataset(
        image_dir=images_dir,
        annotation_file=coco_file,
        split='val',
        split_ratio=args.split_ratio,
        image_size=args.img_size,
        num_classes=config['nc']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create and train model
    model = EnhancedDetectionModel(num_classes=config['nc'])
    model, metrics_history = train_model(model, train_loader, val_loader, config, args.output_dir)

if __name__ == '__main__':
    main()
