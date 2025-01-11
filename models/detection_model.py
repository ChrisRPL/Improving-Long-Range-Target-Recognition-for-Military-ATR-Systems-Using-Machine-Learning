import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet50 but preserve spatial information
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
    def forward(self, x):
        return self.features(x)

class TransformerModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100):
        super().__init__()
        self.d_model = d_model
        
        # Feature projection
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, 32, 32))
        self.query_embed = nn.Parameter(torch.zeros(num_queries, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
    def forward(self, features):
        # Project features
        features = self.input_proj(features)
        
        # Add positional embeddings
        pos = self.pos_embed.repeat(features.shape[0], 1, 1, 1)
        features = features + pos
        
        # Reshape for transformer
        features = features.flatten(2).permute(2, 0, 1)
        
        # Transformer processing
        memory = self.encoder(features)
        tgt = self.query_embed.unsqueeze(1).repeat(1, features.shape[1], 1)
        output = self.decoder(tgt, memory)
        
        return output

class EnhancedDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.transformer = TransformerModel()
        
        # Prediction heads
        self.bbox_head = nn.Linear(512, 4)
        self.class_head = nn.Linear(512, num_classes)
        
        # Metrics
        self.map_metric = MeanAveragePrecision(
            box_format='cxcywh',  # YOLOv8 format
            iou_thresholds=[0.5],
            rec_thresholds=None,
            max_detection_thresholds=[100]
        )
        
    def forward(self, images, flows):
        # Extract features
        img_features = self.backbone(images)
        flow_features = self.backbone(flows)
        
        # Concatenate features along channel dimension
        features = torch.cat([img_features, flow_features], dim=1)
        
        # Transform features
        transformer_output = self.transformer(features)
        
        # Predictions
        boxes = self.bbox_head(transformer_output).sigmoid()  # Normalize to [0,1]
        logits = self.class_head(transformer_output)
        
        return boxes, logits
    
    def loss_fn(self, pred_boxes, pred_logits, target_boxes, target_labels):
        """Calculate combined loss."""
        # Box loss (GIoU Loss)
        giou_loss = 1 - torch.diag(box_iou(
            pred_boxes, target_boxes, 
            box_format='cxcywh'
        )).mean()
        
        # Classification loss
        cls_loss = F.cross_entropy(pred_logits, target_labels)
        
        return giou_loss + cls_loss
    
    def update_metrics(self, pred_boxes, pred_logits, targets):
        """Update metrics for evaluation."""
        predictions = [{
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
        } for boxes, scores, labels in zip(
            pred_boxes, 
            pred_logits.softmax(-1).max(-1).values,
            pred_logits.argmax(-1)
        )]
        
        targets = [{
            'boxes': target['boxes'],
            'labels': target['labels']
        } for target in targets]
        
        self.map_metric.update(predictions, targets)
    
    def get_metrics(self):
        """Calculate and return all metrics."""
        metrics = self.map_metric.compute()
        return {
            'mAP50': metrics['map_50'].item(),
            'mAP50-95': metrics['map'].item(),
            'precision': metrics['map_per_class'].mean().item(),
            'recall': metrics['mar_100'].item()
        }
