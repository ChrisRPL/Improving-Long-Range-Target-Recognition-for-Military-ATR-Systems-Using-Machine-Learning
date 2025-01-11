import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse
import sys
from models.detection_model import EnhancedDetectionModel
from models.detection_dataset import EnhancedObjectDetectionDataset

sys.path.append(str(Path(__file__).parent.parent))

def load_yaml(yaml_path):
    """Load YAML configuration file and resolve paths."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute
    yaml_dir = Path(yaml_path).parent
    if '../' in str(config['train']):
        config['train'] = str(yaml_dir / config['train'])
    if '../' in str(config['val']):
        config['val'] = str(yaml_dir / config['val'])
    if '../' in str(config['test']):
        config['test'] = str(yaml_dir / config['test'])
    
    return config

def train_model(model, train_loader, val_loader, config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4))
    
    # Initialize metric trackers
    best_map50 = 0
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'mAP50': [],
        'mAP50-95': [],
        'precision': [],
        'recall': []
    }
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            flows = batch['flow'].to(device)
            labels = batch['labels'].to(device)
            
            # Get non-empty label masks
            valid_labels = (labels.sum(dim=2) > 0)
            
            # Forward pass
            pred_boxes, pred_logits = model(images, flows)
            
            # Calculate loss
            loss = model.loss_fn(
                pred_boxes, 
                pred_logits, 
                labels[valid_labels][:, 1:],
                labels[valid_labels][:, 0].long()
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Format predictions for metric update
            predictions = [{
                'boxes': boxes,
                'scores': scores,
                'labels': lbls,
            } for boxes, scores, lbls in zip(
                pred_boxes.detach(),
                torch.softmax(pred_logits.detach(), dim=-1).max(dim=-1).values,
                torch.argmax(pred_logits.detach(), dim=-1)
            )]
            
            # Format targets for metric update
            targets = [{
                'boxes': label[label[:, 0] >= 0][:, 1:],
                'labels': label[label[:, 0] >= 0][:, 0].long(),
            } for label in labels]
            
            model.map_metric.update(predictions, targets)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_metrics = model.map_metric.compute()
        model.map_metric.reset()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(device)
                flows = batch['flow'].to(device)
                labels = batch['labels'].to(device)
                valid_labels = (labels.sum(dim=2) > 0)
                
                # Forward pass
                pred_boxes, pred_logits = model(images, flows)
                
                # Calculate loss
                loss = model.loss_fn(
                    pred_boxes,
                    pred_logits,
                    labels[valid_labels][:, 1:],
                    labels[valid_labels][:, 0].long()
                )
                
                val_loss += loss.item()
                
                # Update metrics
                predictions = [{
                    'boxes': boxes,
                    'scores': scores,
                    'labels': lbls,
                } for boxes, scores, lbls in zip(
                    pred_boxes,
                    torch.softmax(pred_logits, dim=-1).max(dim=-1).values,
                    torch.argmax(pred_logits, dim=-1)
                )]
                
                targets = [{
                    'boxes': label[label[:, 0] >= 0][:, 1:],
                    'labels': label[label[:, 0] >= 0][:, 0].long(),
                } for label in labels]
                
                model.map_metric.update(predictions, targets)
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_metrics = model.map_metric.compute()
        model.map_metric.reset()
        
        # Update metrics history
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['mAP50'].append(val_metrics['map_50'].item())
        metrics_history['mAP50-95'].append(val_metrics['map'].item())
        metrics_history['precision'].append(val_metrics['map_per_class'].mean().item())
        metrics_history['recall'].append(val_metrics['mar_100'].item())
        
        # Print epoch metrics
        print(f"\nEpoch {epoch+1} metrics:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"mAP50: {val_metrics['map_50']:.4f}")
        print(f"mAP50-95: {val_metrics['map']:.4f}")
        print(f"Precision: {val_metrics['map_per_class'].mean():.4f}")
        print(f"Recall: {val_metrics['mar_100']:.4f}")
        
        # Save best model
        if val_metrics['map_50'] > best_map50:
            best_map50 = val_metrics['map_50']
            checkpoint_path = Path(output_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config,
            }, str(checkpoint_path))
            print(f"Saved best model with mAP50: {best_map50:.4f}")
        
        # Save metrics history
        metrics_path = Path(output_dir) / 'metrics_history.pth'
        torch.save(metrics_history, str(metrics_path))

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Detection Model')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, default=None,
                      help='Path to pretrained weights (optional)')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='runs/train',
                      help='Output directory')
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.data)
    
    # Update config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
    })
    
    # Print configuration
    print("\nConfiguration:")
    print(f"Dataset path: {config['train']}")
    print(f"Number of classes: {config['nc']}")
    print(f"Class names: {config['names']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}\n")
    
    # Create datasets
    train_dataset = EnhancedObjectDetectionDataset(
        Path(config['train']).parent.parent,  # Get root path
        'train'
    )
    val_dataset = EnhancedObjectDetectionDataset(
        Path(config['val']).parent.parent,  # Get root path
        'valid'
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
    
    # Create model
    model = EnhancedDetectionModel(num_classes=config['nc'])
    
    # Load pretrained weights if provided
    if args.weights:
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train
    train_model(model, train_loader, val_loader, config, args.output_dir)

if __name__ == '__main__':
    main()
