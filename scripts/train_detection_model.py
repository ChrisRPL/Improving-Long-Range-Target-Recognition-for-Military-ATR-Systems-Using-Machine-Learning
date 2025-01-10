import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import sys
from models.detection_dataset import EnhancedObjectDetectionDataset
from models.detection_model import EnhancedDetectionModel

sys.path.append(str(Path(__file__).parent.parent))

def train_model(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    # Training loop
    best_map = 0
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
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
        print(f'Epoch {epoch+1} metrics:')
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')
        
        # Save best model
        if metrics['mAP50'] > best_map:
            best_map = metrics['mAP50']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, 'best_model.pt')

def main():
    # Load config
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = EnhancedObjectDetectionDataset(
        Path(config['path']), 'train'
    )
    val_dataset = EnhancedObjectDetectionDataset(
        Path(config['path']), 'valid'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Create model
    model = EnhancedDetectionModel(num_classes=config['nc'])
    
    # Train
    train_model(model, train_loader, val_loader, config)

if __name__ == '__main__':
    main()
