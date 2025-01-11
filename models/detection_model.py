import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.detection import MeanAveragePrecision
import math

class Backbone(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Enable gradient checkpointing
        resnet.layer1.use_checkpoint = True
        resnet.layer2.use_checkpoint = True
        resnet.layer3.use_checkpoint = True
        resnet.layer4.use_checkpoint = True
        
        if input_channels != 3:
            self.first_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            pretrained_weight = resnet.conv1.weight.data
            self.first_conv.weight.data = torch.mean(pretrained_weight, dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
        else:
            self.first_conv = resnet.conv1
        
        self.features = nn.Sequential(
            self.first_conv,
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
        
        # Feature projection (adjust input channels for concatenated features)
        self.input_proj = nn.Conv2d(4096, d_model, kernel_size=1)  # 2048*2 because we concatenate two backbone outputs
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, 32, 32))
        self.query_embed = nn.Parameter(torch.zeros(num_queries, d_model))
        
        # Transformer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=2048,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=2048,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Initialize positional embeddings
        torch.nn.init.normal_(self.pos_embed, mean=0, std=0.02)
        torch.nn.init.normal_(self.query_embed, mean=0, std=0.02)
        
    def forward(self, features):
        # Project features
        features = self.input_proj(features)  # [B, d_model, H, W]
        
        # Add positional embeddings
        pos = self.pos_embed.repeat(features.shape[0], 1, 1, 1)
        features = features + pos
        
        # Reshape for transformer (batch_first=True)
        features = features.flatten(2).transpose(1, 2)  # [B, HW, d_model]
        
        # Transformer processing
        memory = self.encoder(features)
        
        # Prepare queries
        batch_size = features.shape[0]
        tgt = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, d_model]
        
        # Decode
        output = self.decoder(tgt, memory)
        
        return output

class EnhancedDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Separate backbones for image and flow
        self.img_backbone = Backbone(input_channels=3)
        self.flow_backbone = Backbone(input_channels=2)
        
        # Transformer
        self.transformer = TransformerModel()
        
        # Prediction heads
        self.bbox_head = nn.Linear(512, 4)  # (cx, cy, w, h)
        self.class_head = nn.Linear(512, num_classes)
        
        # Metrics
        self.map_metric = MeanAveragePrecision(
            box_format='cxcywh',  # YOLOv8 format
            iou_thresholds=[0.5],
            rec_thresholds=None,
            max_detection_thresholds=[1, 10, 100]
        )
    
    def forward(self, images, flows):
        """
        Forward pass of the model.
        Args:
            images: Tensor of shape [B, 3, H, W]
            flows: Tensor of shape [B, 2, H, W]
        Returns:
            boxes: Tensor of shape [B, num_queries, 4] in (cx, cy, w, h) format
            logits: Tensor of shape [B, num_queries, num_classes]
        """
        # Extract features using separate backbones
        img_features = self.img_backbone(images)  # [B, 2048, H/32, W/32]
        flow_features = self.flow_backbone(flows)  # [B, 2048, H/32, W/32]
        
        # Concatenate features along channel dimension
        features = torch.cat([img_features, flow_features], dim=1)  # [B, 4096, H/32, W/32]
        
        # Transform features
        transformer_output = self.transformer(features)  # [B, num_queries, d_model]
        
        # Predictions
        boxes = self.bbox_head(transformer_output).sigmoid()  # Normalize to [0,1]
        logits = self.class_head(transformer_output)
        
        return boxes, logits
    
    def loss_fn(self, pred_boxes, pred_logits, target_boxes, target_labels):
        """
        Calculate loss for training.
        Args:
            pred_boxes: Predicted boxes [B*num_queries, 4]
            pred_logits: Predicted class logits [B*num_queries, num_classes]
            target_boxes: Ground truth boxes [N, 4]
            target_labels: Ground truth labels [N]
        Returns:
            total_loss: Combined loss for training
        """
        # Calculate giou loss for boxes
        giou_loss = 1 - torch.diag(box_iou(
            pred_boxes,
            target_boxes,
            box_format='cxcywh'
        )).mean()
        
        # Calculate classification loss
        cls_loss = nn.functional.cross_entropy(pred_logits, target_labels)
        
        # Combine losses
        total_loss = giou_loss + cls_loss
        
        return total_loss
    
    def update_metrics(self, pred_boxes, pred_logits, targets):
        """
        Update evaluation metrics.
        Args:
            pred_boxes: Predicted boxes [B, num_queries, 4]
            pred_logits: Predicted class logits [B, num_queries, num_classes]
            targets: List of target dictionaries containing 'boxes' and 'labels'
        """
        predictions = [{
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
        } for boxes, scores, labels in zip(
            pred_boxes,
            torch.softmax(pred_logits, dim=-1).max(dim=-1).values,
            torch.argmax(pred_logits, dim=-1)
        )]
        
        self.map_metric.update(predictions, targets)
    
    def get_metrics(self):
        """
        Compute and return all metrics.
        Returns:
            metrics: Dictionary containing all computed metrics
        """
        metrics = self.map_metric.compute()
        return {
            'mAP50': metrics['map_50'].item(),
            'mAP50-95': metrics['map'].item(),
            'precision': metrics['map_per_class'].mean().item(),
            'recall': metrics['mar_100'].item()
        }
