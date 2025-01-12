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

sys.path.append(str(Path(__file__).parent.parent))

def setup_logger(output_dir):
    """Setup logging configuration"""
    log_file = Path(output_dir) / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Setup stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def load_yaml(yaml_path):
    """Load YAML configuration file and resolve paths."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute
    yaml_dir = Path(yaml_path).parent
    if '../' in str(config.get('train', '')):
        config['train'] = str(yaml_dir / config['train'])
    if '../' in str(config.get('val', '')):
        config['val'] = str(yaml_dir / config['val'])
    if '../' in str(config.get('test', '')):
        config['test'] = str(yaml_dir / config['test'])
    
    return config

def collate_fn(batch):
    """
    Custom collate function to handle batches.
    """
    images = torch.stack([item['image'] for item in batch])
    flows = torch.stack([item['flow'] for item in batch])
    
    # Get max number of labels in batch
    max_labels = max(item['labels'].shape[0] for item in batch)
    
    # Pad labels to same size
    padded_labels = []
    for item in batch:
        labels = item['labels']
        if labels.shape[0] < max_labels:
            padding = torch.zeros((max_labels - labels.shape[0], 5), dtype=labels.dtype)
            labels = torch.cat([labels, padding], dim=0)
        padded_labels.append(labels)
    
    labels = torch.stack(padded_labels)
    
    return {
        'image': images,
        'flow': flows,
        'labels': labels,
        'img_path': [item['img_path'] for item in batch]
    }

def train_model(model, train_loader, val_loader, config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    logger = setup_logger(output_dir)
    model = model.to(device)
    
    # Setup optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Initialize the grad scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    # Initialize metrics
    best_map50 = 0
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'mAP50': [],
        'mAP50-95': [],
        'precision': [],
        'recall': []
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['epochs']):
        logger.info(f"\nStarting epoch {epoch+1}/{config['epochs']}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        # Reset metrics for new epoch
        model.map_metric.reset()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(device)
                flows = batch['flow'].to(device)
                labels = batch['labels'].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    pred_boxes, pred_logits = model(images, flows)
                    loss = model.loss_fn(
                        pred_boxes,
                        pred_logits,
                        labels[..., 1:],  # box coordinates
                        labels[..., 0]    # class labels
                    )
                
                # Backward pass and optimization
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Update training metrics
                target_list = []
                for i in range(len(labels)):
                    valid_mask = labels[i, :, 0] >= 0
                    target_list.append({
                        'boxes': labels[i, valid_mask, 1:].cpu(),
                        'labels': labels[i, valid_mask, 0].long().cpu()
                    })
                
                pred_list = []
                for i in range(len(pred_boxes)):
                    scores = torch.softmax(pred_logits[i], dim=-1)[:, 1:]  # Remove background class
                    pred_list.append({
                        'boxes': pred_boxes[i].detach().cpu(),
                        'scores': scores.max(dim=-1)[0].detach().cpu(),
                        'labels': scores.argmax(dim=-1).detach().cpu()
                    })
                
                model.map_metric.update(pred_list, target_list)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate epoch metrics
        train_metrics = model.map_metric.compute()
        model.map_metric.reset()
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        logger.info("Starting validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                flows = batch['flow'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.cuda.amp.autocast():
                    pred_boxes, pred_logits = model(images, flows)
                    loss = model.loss_fn(
                        pred_boxes,
                        pred_logits,
                        labels[..., 1:],
                        labels[..., 0]
                    )
                
                val_loss += loss.item()
                
                # Format validation predictions and targets
                for i in range(len(labels)):
                    valid_mask = labels[i, :, 0] >= 0
                    val_targets.append({
                        'boxes': labels[i, valid_mask, 1:].cpu(),
                        'labels': labels[i, valid_mask, 0].long().cpu()
                    })
                    
                    scores = torch.softmax(pred_logits[i], dim=-1)[:, 1:]
                    val_preds.append({
                        'boxes': pred_boxes[i].cpu(),
                        'scores': scores.max(dim=-1)[0].cpu(),
                        'labels': scores.argmax(dim=-1).cpu()
                    })
        
        # Compute validation metrics
        model.map_metric.update(val_preds, val_targets)
        val_metrics = model.map_metric.compute()
        model.map_metric.reset()
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"mAP50: {val_metrics['map_50']:.4f}")
        logger.info(f"mAP50-95: {val_metrics['map']:.4f}")
        logger.info(f"Precision: {val_metrics['map_per_class'].mean():.4f}")
        logger.info(f"Recall: {val_metrics['mar_100']:.4f}")
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save metrics history
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['mAP50'].append(val_metrics['map_50'].item())
        metrics_history['mAP50-95'].append(val_metrics['map'].item())
        metrics_history['precision'].append(val_metrics['map_per_class'].mean().item())
        metrics_history['recall'].append(val_metrics['mar_100'].item())
        
        # Save best model
        if val_metrics['map_50'] > best_map50:
            best_map50 = val_metrics['map_50']
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, str(checkpoint_path))
            logger.info(f"Saved new best model with mAP50: {best_map50:.4f}")
        
        # Save latest model
        checkpoint_path = output_dir / 'last_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'metrics': val_metrics,
            'config': config,
        }, str(checkpoint_path))
        
        # Save metrics history
        torch.save(metrics_history, str(output_dir / 'metrics_history.pt'))
        
        # Plot metrics
        plot_metrics(metrics_history, output_dir / 'metrics.png')
    
    return model, metrics_history

def plot_metrics(metrics_history, save_path):
    """Plot training metrics history."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(metrics_history['train_loss'], label='Train Loss')
        plt.plot(metrics_history['val_loss'], label='Val Loss')
        plt.title('Losses')
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
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        
        # Plot recall
        plt.subplot(2, 2, 4)
        plt.plot(metrics_history['recall'])
        plt.title('Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Detection Model')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, default=None,
                      help='Path to pretrained weights (optional)')
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
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load and print config
    config = load_yaml(args.data)
    
    # Update config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'img_size': args.img_size,
        'device': args.device
    })
    
    # Print configuration
    print("\nConfiguration:")
    print(f"Dataset path: {config['train']}")
    print(f"Number of classes: {config['nc']}")
    print(f"Class names: {config['names']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Image size: {config['img_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Device: {config['device']}\n")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create datasets
        train_dataset = EnhancedObjectDetectionDataset(
            Path(config['train']).parent.parent,
            'train',
            image_size=args.img_size
        )
        print(f"Created training dataset with {len(train_dataset)} samples")
        
        val_dataset = EnhancedObjectDetectionDataset(
            Path(config['val']).parent.parent,
            'valid',
            image_size=args.img_size
        )
        print(f"Created validation dataset with {len(val_dataset)} samples")
        
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
        
        # Create model
        print("Creating model...")
        model = EnhancedDetectionModel(num_classes=config['nc'])
        model = model.to(device)
        
        # Load pretrained weights if provided
        if args.weights:
            print(f"Loading weights from {args.weights}")
            checkpoint = torch.load(args.weights, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Weights loaded successfully")
        
        # Train
        print("\nStarting training...")
        model, metrics_history = train_model(model, train_loader, val_loader, config, args.output_dir)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
