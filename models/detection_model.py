import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
from torchvision.ops import box_iou as tv_box_iou
from scipy.optimize import linear_sum_assignment
import math


class Backbone(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if input_channels != 3:
            self.first_conv = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            pretrained_weight = resnet.conv1.weight.data
            self.first_conv.weight.data = torch.mean(
                pretrained_weight, dim=1, keepdim=True
            ).repeat(1, input_channels, 1, 1)
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
            resnet.layer4,
        )

    def forward(self, x):
        return self.features(x)  # Output: [B, 2048, H/32, W/32]


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_queries=100,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Input projection for backbone features
        self.input_proj = nn.Conv2d(4096, d_model, kernel_size=1)

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, d_model, 13, 13)
        )  # For 416x416 input
        self.query_embed = nn.Parameter(torch.zeros(num_queries, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True
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
        pos = F.interpolate(
            self.pos_embed, size=(H, W), mode="bicubic", align_corners=True
        )
        features = features + pos

        # Reshape for transformer
        features = features.flatten(2).transpose(1, 2)  # [B, H*W, d_model]

        # Generate queries
        queries = self.query_embed.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, num_queries, d_model]

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
            d_model=self.d_model, num_queries=self.num_queries
        )

        # Prediction heads
        self.bbox_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, 4)
        )

        self.class_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, num_classes + 1),  # +1 for background class
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
            box_format="cxcywh",
            iou_thresholds=[0.5],
            rec_thresholds=None,
            max_detection_thresholds=[1, 10, 100],
        )

    def forward(self, images, flows):
        # Extract features [B, 2048, H/32, W/32]
        img_features = self.img_backbone(images)
        flow_features = self.flow_backbone(flows)
    
        print(f"\nBackbone features range:")
        print(f"Image features: [{img_features.min():.4f}, {img_features.max():.4f}]")
        print(f"Flow features: [{flow_features.min():.4f}, {flow_features.max():.4f}]")
    
        # Concatenate features
        features = torch.cat([img_features, flow_features], dim=1)
    
        # Transform features
        transformer_output = self.transformer(features)
    
        print(f"Transformer output range: [{transformer_output.min():.4f}, {transformer_output.max():.4f}]")
    
        # Predict boxes and classes
        boxes = self.bbox_head(transformer_output).sigmoid()  # [B, num_queries, 4]
        logits = self.class_head(transformer_output)  # [B, num_queries, num_classes+1]
    
        print(f"Output ranges:")
        print(f"Boxes (should be [0,1]): [{boxes.min():.4f}, {boxes.max():.4f}]")
        print(f"Logits: [{logits.min():.4f}, {logits.max():.4f}]")
    
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

        boxes1_xyxy = torch.stack(
            [x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2], dim=-1
        )

        boxes2_xyxy = torch.stack(
            [x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2], dim=-1
        )

        return tv_box_iou(boxes1_xyxy, boxes2_xyxy)

    def loss_fn(self, pred_boxes, pred_logits, target_boxes, target_labels):
        B = pred_boxes.shape[0]
        device = pred_boxes.device
    
        # Initialize loss
        batch_total_loss = torch.tensor(0., device=device, requires_grad=True)
        num_boxes = 0
    
        for i in range(B):
            # Check if we have any valid targets (non-dummy)
            valid_mask = (target_labels[i] >= 0) & (target_labels[i] < self.num_classes)
            if not valid_mask.any():
                # Add small loss to maintain gradient flow
                dummy_loss = pred_boxes[i].sum() * 0.0 + pred_logits[i].sum() * 0.0
                batch_total_loss = batch_total_loss + dummy_loss
                continue
            
            batch_target_boxes = target_boxes[i][valid_mask]
            batch_target_labels = target_labels[i][valid_mask]
        
            # Calculate IoU matrix
            iou_matrix = self.box_iou(pred_boxes[i], batch_target_boxes)
        
            # Get best matches
            values, indices = iou_matrix.max(dim=1)
        
            # Calculate losses only for matched predictions
            matched_boxes = pred_boxes[i][values > 0.5]  # IoU threshold
            matched_targets = batch_target_boxes[indices[values > 0.5]]
        
            if len(matched_boxes) > 0:
                # Box regression loss
                box_loss = F.l1_loss(matched_boxes, matched_targets, reduction='mean')
            
                # Classification loss
                target_classes = torch.full((pred_logits[i].shape[0],), 
                                     self.num_classes,  # background class
                                     device=device)
                target_classes[values > 0.5] = batch_target_labels[indices[values > 0.5]]
                cls_loss = F.cross_entropy(pred_logits[i], target_classes)
            
                # Combined loss
                loss = box_loss * 5.0 + cls_loss
                batch_total_loss = batch_total_loss + loss
                num_boxes += len(matched_boxes)
    
        # Return average loss
        if num_boxes > 0:
            return batch_total_loss / num_boxes
        else:
            # Add small loss to maintain gradient flow
            return batch_total_loss + pred_boxes.sum() * 0.0 + pred_logits.sum() * 0.0
    
    def generalized_box_iou(self, boxes1, boxes2):
        """
        Compute generalized IoU between two sets of boxes.
        """
        # Convert boxes to xyxy format
        x1, y1, w1, h1 = boxes1.unbind(-1)
        x2, y2, w2, h2 = boxes2.unbind(-1)
    
        b1_x1, b1_y1 = x1 - w1/2, y1 - h1/2
        b1_x2, b1_y2 = x1 + w1/2, y1 + h1/2
        b2_x1, b2_y1 = x2 - w2/2, y2 - h2/2
        b2_x2, b2_y2 = x2 + w2/2, y2 + h2/2
    
        # Calculate areas
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
        # Calculate intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
    
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h
    
        # Calculate union
        union = area1 + area2 - intersection
    
        # Calculate IoU
        iou = intersection / union.clamp(min=1e-6)
    
        # Calculate enclosing box
        encl_x1 = torch.min(b1_x1, b2_x1)
        encl_y1 = torch.min(b1_y1, b2_y1)
        encl_x2 = torch.max(b1_x2, b2_x2)
        encl_y2 = torch.max(b1_y2, b2_y2)
    
        encl_w = (encl_x2 - encl_x1).clamp(min=0)
        encl_h = (encl_y2 - encl_y1).clamp(min=0)
        enclosure = encl_w * encl_h
    
        # Calculate GIoU
        giou = iou - (enclosure - union) / enclosure.clamp(min=1e-6)
    
        return giou

    def update_metrics(self, pred_boxes, pred_logits, targets):
        """Update evaluation metrics"""
        predictions = [
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
            for boxes, scores, labels in zip(
                pred_boxes,
                torch.softmax(pred_logits, dim=-1).max(dim=-1).values,
                torch.argmax(pred_logits, dim=-1),
            )
        ]

        self.map_metric.update(predictions, targets)

    def get_metrics(self):
        """Get computed metrics"""
        metrics = self.map_metric.compute()
        return {
            "mAP50": metrics["map_50"].item(),
            "mAP50-95": metrics["map"].item(),
            "precision": metrics["map_per_class"].mean().item(),
            "recall": metrics["mar_100"].item(),
        }

