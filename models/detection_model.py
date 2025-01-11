import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
from torchvision.ops import box_iou as tv_box_iou
import math

class Backbone(nn.Module):
   def __init__(self, input_channels=3):
       super().__init__()
       resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
       
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
       return self.features(x)  # Output: [B, 2048, H/32, W/32]

class TransformerModel(nn.Module):
   def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100):
       super().__init__()
       self.d_model = d_model
       self.num_queries = num_queries
       
       # Input projection for backbone features
       self.input_proj = nn.Conv2d(4096, d_model, kernel_size=1)
       
       # Positional embeddings
       self.pos_embed = nn.Parameter(torch.zeros(1, d_model, 13, 13))  # For 416x416 input
       self.query_embed = nn.Parameter(torch.zeros(num_queries, d_model))
       
       # Transformer
       encoder_layer = nn.TransformerEncoderLayer(
           d_model=d_model, 
           nhead=nhead,
           dim_feedforward=2048,
           batch_first=True
       )
       
       decoder_layer = nn.TransformerDecoderLayer(
           d_model=d_model,
           nhead=nhead,
           dim_feedforward=2048,
           batch_first=True
       )
       
       self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
       self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
       
       # Initialize embeddings
       nn.init.normal_(self.pos_embed, mean=0, std=0.02)
       nn.init.normal_(self.query_embed, mean=0, std=0.02)
       
   def forward(self, features):
       # Project concatenated features 
       features = self.input_proj(features)  # [B, d_model, H, W]
       B, C, H, W = features.shape
       
       # Add positional encoding
       pos = F.interpolate(self.pos_embed, size=(H, W), mode='bicubic', align_corners=True)
       features = features + pos
       
       # Reshape for transformer
       features = features.flatten(2).transpose(1, 2)  # [B, H*W, d_model]
       
       # Generate queries 
       queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, d_model]
       
       # Transformer processing 
       memory = self.encoder(features)
       output = self.decoder(queries, memory)  # [B, num_queries, d_model]
       
       return output

class EnhancedDetectionModel(nn.Module):
   def __init__(self, num_classes, num_queries=100):
       super().__init__()
       self.num_classes = num_classes
       self.d_model = 256
       self.num_queries = num_queries
       
       # Backbones
       self.img_backbone = Backbone(input_channels=3)
       self.flow_backbone = Backbone(input_channels=2)
       
       # Transformer
       self.transformer = TransformerModel(
           d_model=self.d_model,
           num_queries=self.num_queries
       )
       
       # Prediction heads
       self.bbox_head = nn.Sequential(
           nn.Linear(self.d_model, self.d_model),
           nn.ReLU(),
           nn.Linear(self.d_model, 4)
       )
       
       self.class_head = nn.Sequential(
           nn.Linear(self.d_model, self.d_model),
           nn.ReLU(),
           nn.Linear(self.d_model, num_classes + 1)  # +1 for background class
       )
       
       # Initialize prediction heads
       for layer in self.bbox_head.modules():
           if isinstance(layer, nn.Linear):
               nn.init.xavier_uniform_(layer.weight)
               nn.init.zeros_(layer.bias)
               
       for layer in self.class_head.modules():
           if isinstance(layer, nn.Linear):
               nn.init.xavier_uniform_(layer.weight)
               nn.init.zeros_(layer.bias)
       
       # Metrics
       self.map_metric = MeanAveragePrecision(
           box_format='cxcywh',
           iou_thresholds=[0.5],
           rec_thresholds=None,
           max_detection_thresholds=[1, 10, 100]
       )
   
   def forward(self, images, flows):
       """
       Forward pass
       Args:
           images: [B, 3, H, W] 
           flows: [B, 2, H, W]
       Returns:
           boxes: [B, num_queries, 4]
           logits: [B, num_queries, num_classes+1]  
       """
       # Extract features [B, 2048, H/32, W/32]
       img_features = self.img_backbone(images)
       flow_features = self.flow_backbone(flows)
       
       # Concatenate features
       features = torch.cat([img_features, flow_features], dim=1)  # [B, 4096, H/32, W/32]
       
       # Transform features
       transformer_output = self.transformer(features)  # [B, num_queries, d_model]
       
       # Predict boxes and classes
       boxes = self.bbox_head(transformer_output).sigmoid()  # [B, num_queries, 4]
       logits = self.class_head(transformer_output)  # [B, num_queries, num_classes+1]
       
       return boxes, logits
   
   def box_iou(self, boxes1, boxes2):
       """
       Calculate IoU between boxes in cxcywh format
       Args:
           boxes1, boxes2: [..., 4] in cxcywh format
       Returns:
           iou: [...] IoU values
       """
       # Convert to xyxy format
       x1, y1, w1, h1 = boxes1.unbind(-1)
       x2, y2, w2, h2 = boxes2.unbind(-1)
       
       boxes1_xyxy = torch.stack([
           x1 - w1/2, y1 - h1/2,
           x1 + w1/2, y1 + h1/2
       ], dim=-1)
       
       boxes2_xyxy = torch.stack([
           x2 - w2/2, y2 - h2/2,
           x2 + w2/2, y2 + h2/2
       ], dim=-1)
       
       return tv_box_iou(boxes1_xyxy, boxes2_xyxy)

   def loss_fn(self, pred_boxes, pred_logits, target_boxes, target_labels):
       """
       Calculate loss
       Args:
           pred_boxes: [B, num_queries, 4] predicted boxes
           pred_logits: [B, num_queries, num_classes+1] predicted logits
           target_boxes: [B, max_objects, 4] ground truth boxes 
           target_labels: [B, max_objects] ground truth labels
       Returns:
           total_loss: scalar loss value
       """
       B = pred_boxes.shape[0]
       device = pred_boxes.device
       total_loss = torch.tensor(0.0, device=device)

       # Verify shapes
       assert pred_boxes.shape[0] == pred_logits.shape[0] == target_boxes.shape[0] == target_labels.shape[0]
       
       for i in range(B):
           # Get valid targets (remove padding)
           valid_mask = target_labels[i] >= 0
           batch_target_boxes = target_boxes[i][valid_mask]
           batch_target_labels = target_labels[i][valid_mask]
           
           if len(batch_target_boxes) == 0:
               # Handle empty targets
               total_loss = total_loss + pred_boxes[i].sum() * 0.0
               continue
           
           # Classification loss with padded targets
           num_valid_targets = len(batch_target_labels)
           padded_labels = torch.cat([
               batch_target_labels,
               torch.zeros(self.num_queries - num_valid_targets, device=device)
           ]).long()
           
           cls_loss = F.cross_entropy(pred_logits[i], padded_labels)
           
           # Box loss (GIoU)
           valid_pred_boxes = pred_boxes[i][:num_valid_targets]
           if len(batch_target_boxes) > 0:
               iou = self.box_iou(valid_pred_boxes, batch_target_boxes)
               max_iou, _ = iou.max(dim=1)
               box_loss = (1 - max_iou).mean()
           else:
               box_loss = valid_pred_boxes.sum() * 0.0
           
           total_loss = total_loss + (cls_loss + box_loss)
       
       return total_loss / B

   def update_metrics(self, pred_boxes, pred_logits, targets):
       """Update evaluation metrics"""
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
       """Get computed metrics"""
       metrics = self.map_metric.compute()
       return {
           'mAP50': metrics['map_50'].item(),
           'mAP50-95': metrics['map'].item(),
           'precision': metrics['map_per_class'].mean().item(),
           'recall': metrics['mar_100'].item()
       }
