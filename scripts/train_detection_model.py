import torch
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import wandb
import json

from models.detection_model import EnhancedDetectionModel
from models.detection_dataset import DataModule

import sys

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

def visualize_batch(images, boxes, labels, category_names, output_dir, epoch, batch_idx):
    """Visualize a batch of predictions"""
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, (img, bbox, label) in enumerate(zip(images[:8], boxes[:8], labels[:8])):
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        axes[idx].imshow(img)
        
        for box, lbl in zip(bbox, label):
            if lbl == 0:  # Skip background
                continue
            
            x, y, w, h = box.cpu().numpy()
            rect = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            axes[idx].add_patch(rect)
            axes[idx].text(
                x - w/2, y - h/2 - 2,
                category_names.get(lbl.item(), 'Unknown'),
                color='white', bbox=dict(facecolor='red', alpha=0.5)
            )
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'batch_viz_epoch{epoch}_batch{batch_idx}.png')
    plt.close()

def train_model(args):
    """Main training function"""
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir)
    logger.info(f"Starting training with args: {args}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="enhanced-detection", config=args)
    
    # Setup data
    data_module = DataModule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        split_ratio=args.split_ratio,
        augment=True
    )
    data_module.setup()
    
    category_names = data_module.get_category_names()
    
    # Create model
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedDetectionModel(num_classes=data_module.num_classes)
    model = model.to(device_type)
    
    # Save category mapping
    with open(output_dir / 'category_mapping.json', 'w') as f:
        json.dump(category_names, f, indent=2)
    
    # Setup training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scaler = GradScaler(enabled=True)
    
    # Calculate total steps for scheduler
    total_steps = len(data_module.train_dataloader()) * args.epochs
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_map50 = 0
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'mAP50': [], 'mAP50-95': [],
        'precision': [], 'recall': [],
        'per_class_map': []
    }
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        model.map_metric.reset()
        
        train_loader = data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device_type)
            flows = batch['flow'].to(device_type)
            boxes = batch['boxes'].to(device_type)
            labels = batch['labels'].to(device_type)
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast('cuda'):
                pred_boxes, pred_logits = model(images, flows)
                loss = model.loss_fn(pred_boxes, pred_logits, boxes, labels)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            model.update_metrics(pred_boxes.detach(), pred_logits.detach(), boxes, labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Visualize batch occasionally
            if batch_idx % args.viz_interval == 0:
                visualize_batch(
                    images.detach(), boxes.detach(), labels.detach(),
                    category_names, output_dir, epoch, batch_idx
                )
        
        # Calculate training metrics
        train_metrics = model.get_metrics()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        model.map_metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(data_module.val_dataloader(), desc="Validation"):
                images = batch['image'].to(device_type)
                flows = batch['flow'].to(device_type)
                boxes = batch['boxes'].to(device_type)
                labels = batch['labels'].to(device_type)
                
                pred_boxes, pred_logits = model(images, flows)
                loss = model.loss_fn(pred_boxes, pred_logits, boxes, labels)
                
                val_loss += loss.item()
                model.update_metrics(pred_boxes, pred_logits, boxes, labels)
        
        # Calculate validation metrics
        val_metrics = model.get_metrics()
        avg_val_loss = val_loss / len(data_module.val_dataloader())
        
        # Update metrics history
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['mAP50'].append(val_metrics['mAP50'])
        metrics_history['mAP50-95'].append(val_metrics['mAP50-95'])
        metrics_history['precision'].append(val_metrics['precision'])
        metrics_history['recall'].append(val_metrics['recall'])
        metrics_history['per_class_map'].append(val_metrics['per_class_map'])
        
        # Log metrics
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"mAP50: {val_metrics['mAP50']:.4f}")
        logger.info(f"mAP50-95: {val_metrics['mAP50-95']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'mAP50': val_metrics['mAP50'],
                'mAP50-95': val_metrics['mAP50-95'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall']
            })
        
        # Save best model
        if val_metrics['mAP50'] > best_map50:
            best_map50 = val_metrics['mAP50']
            save_path = output_dir / 'best_model.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'args': args,
            }, str(save_path))
            
            logger.info(f"Saved new best model with mAP50: {best_map50:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'args': args,
            }, str(checkpoint_path))
        
        # Save metrics history
        torch.save(metrics_history, output_dir / 'metrics_history.pt')

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Detection Model')
    
    # Dataset arguments
    parser.add_argument('--dataset-dir', type=str, required=True,
                      help='Path to dataset directory containing data/, flow/, and coco.json')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                      help='Train/validation split ratio')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Gradient clipping value')
    
    # Model parameters
    parser.add_argument('--image-size', type=int, default=416,
                      help='Input image size')
    parser.add_argument('--num-queries', type=int, default=100,
                      help='Number of object queries')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers')
    parser.add_argument('--output-dir', type=str, default='runs/train',
                      help='Output directory')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--viz-interval', type=int, default=50,
                      help='Visualize predictions every N batches')
    
    # Additional features
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--use-wandb', action='store_true',
                      help='Use Weights & Biases for logging')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    train_model(args)

if __name__ == '__main__':
    main()
