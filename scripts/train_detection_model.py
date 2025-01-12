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

def collate_fn(batch: List[Dict]) -> Dict:
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

def train_model(model, train_loader, val_loader, config, output_dir):
    # Setup logger
    logger = setup_logger(output_dir)
    logger.info("Starting training with configuration:")
    logger.info(f"Config: {config}")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        # Log model architecture
        logger.debug(f"Model architecture:\n{model}")
        
        # Setup optimizer and scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4))
        scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
        logger.info(f"Optimizer: {optimizer}")
        
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
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    images = batch['image'].to(device)
                    flows = batch['flow'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Log shapes for debugging
                    if batch_idx == 0:
                        print("\nFirst batch samples:")
                        print(f"Images shape: {images.shape}")
                        print(f"Flows shape: {flows.shape}")
                        print(f"Labels shape: {labels.shape}")
                        print(f"Sample labels:\n{labels[0]}")
            
                        # Add additional checks
                        print(f"Label statistics:")
                        print(f"Unique label values: {torch.unique(labels)}")
                        print(f"Number of non-zero labels: {(labels != 0).sum()}")
                    
                    # Zero gradients
                    optimizer.zero_grad(set_to_none=False)
                    
                    # Forward pass with autocast
                    with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                        pred_boxes, pred_logits = model(images, flows)
                        loss = model.loss_fn(
                            pred_boxes,
                            pred_logits,
                            labels[..., 1:],
                            labels[..., 0]
                        )
                    
                    # Log loss components
                    if batch_idx % 50 == 0:
                        logger.info(f"Batch {batch_idx} - Loss: {loss.item():.4f}")
                    
                    # Backward pass and optimization
                    if device.type == 'cuda':
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                        optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    
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
                        scores = torch.softmax(pred_logits[i], dim=-1)[:, 1:]
                        pred_list.append({
                            'boxes': pred_boxes[i].detach().cpu(),
                            'scores': scores.max(dim=-1)[0].detach().cpu(),
                            'labels': scores.argmax(dim=-1).detach().cpu()
                        })
                    
                    model.map_metric.update(pred_list, target_list)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    logger.debug(f"Batch contents: {batch}")
                    raise
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Average training loss for epoch {epoch+1}: {avg_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                logger.info("Starting validation...")
                for batch in tqdm(val_loader, desc="Validation"):
                    images = batch['image'].to(device)
                    flows = batch['flow'].to(device)
                    labels = batch['labels'].to(device)
                    
                    with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
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
            metrics = model.get_metrics()
            model.map_metric.reset()
            
            # Log metrics
            logger.info(f"\nEpoch {epoch+1} Results:")
            logger.info(f"Training Loss: {avg_loss:.4f}")
            logger.info(f"Validation Loss: {val_loss/len(val_loader):.4f}")
            logger.info(f"mAP50: {metrics['mAP50']:.4f}")
            logger.info(f"mAP50-95: {metrics['mAP50-95']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            
            # Save metrics history
            metrics_history['train_loss'].append(avg_loss)
            metrics_history['val_loss'].append(val_loss/len(val_loader))
            metrics_history['mAP50'].append(metrics['mAP50'])
            metrics_history['mAP50-95'].append(metrics['mAP50-95'])
            metrics_history['precision'].append(metrics['precision'])
            metrics_history['recall'].append(metrics['recall'])
            
            # Save best model
            if metrics['mAP50'] > best_map50:
                best_map50 = metrics['mAP50']
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'metrics': metrics,
                    'config': config,
                }, str(checkpoint_path))
                logger.info(f"Saved new best model with mAP50: {best_map50:.4f}")
            
            # Save latest model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'metrics': metrics,
                'config': config,
            }, str(output_dir / 'last_model.pt'))
            
            # Save metrics history
            torch.save(metrics_history, str(output_dir / 'metrics_history.pt'))
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("Training completed successfully")
    return model, metrics_history

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
