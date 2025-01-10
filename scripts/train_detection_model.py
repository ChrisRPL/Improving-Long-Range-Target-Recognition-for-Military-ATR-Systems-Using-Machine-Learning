import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import argparse

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
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    training_params = {
        'lr': config.get('lr', 1e-4),
        'epochs': config.get('epochs', 100),
        'batch_size': config.get('batch_size', 16),
        'weight_decay': config.get('weight_decay', 0.0001)
    }
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_params['lr'],
        weight_decay=training_params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=training_params['lr'],
        epochs=training_params['epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    # Training loop
    best_map = 0
    for epoch in range(training_params['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{training_params["epochs"]}')
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            flows = batch['flow'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            pred_boxes, pred_logits = model(images, flows)
            loss = model.loss_fn(pred_boxes, pred_logits, labels[:, 1:], labels[:, 0])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            model.update_metrics(pred_boxes, pred_logits, labels)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Evaluate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                flows = batch['flow'].to(device)
                labels = batch['labels'].to(device)
                
                pred_boxes, pred_logits = model(images, flows)
                loss = model.loss_fn(pred_boxes, pred_logits, labels[:, 1:], labels[:, 0])
                val_loss += loss.item()
                
                model.update_metrics(pred_boxes, pred_logits, labels)
        
        # Get metrics
        metrics = model.get_metrics()
        print(f'\nEpoch {epoch+1} metrics:')
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')
        
        # Save best model
        if metrics['mAP50'] > best_map:
            best_map = metrics['mAP50']
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config,
            }, str(checkpoint_path))
            print(f"Saved best model with mAP50: {best_map:.4f}")
        
        # Save last model
        checkpoint_path = output_dir / 'last_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
        }, str(checkpoint_path))

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
        num_workers=min(8, config['batch_size']),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=min(8, config['batch_size']),
        pin_memory=True
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
