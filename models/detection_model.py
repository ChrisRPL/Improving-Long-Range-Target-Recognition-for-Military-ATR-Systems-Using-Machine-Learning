import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from scipy.optimize import linear_sum_assignment
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

        # Extract multiple scales from ResNet
        self.layer0 = nn.Sequential(
            self.first_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

    def forward(self, x):
        features = []
        x = self.layer0(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, features):
        results = []
        last_inner = self.inner_blocks[-1](features[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results

class MultiScaleTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # Create position embedding modules properly
        self.pos_embeddings = nn.ModuleDict({
            f'scale_{size}': nn.Parameter(torch.zeros(1, d_model, size, size))
            for size in [64, 32, 16, 8]
        })
        
        # Initialize position embeddings
        for embed in self.pos_embeddings.values():
            nn.init.normal_(embed, mean=0, std=0.02)
        
        # Scale-specific transformers
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model*4,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            for _ in range(4)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(4))
        
    def forward(self, features):
        outputs = []
        
        for idx, (feature, (_, pos_embed), transformer) in enumerate(zip(
            features, 
            self.pos_embeddings.items(), 
            self.transformers
        )):
            B, C, H, W = feature.shape
            
            # Add positional encoding
            feature = feature + F.interpolate(pos_embed, size=(H, W), mode='bilinear', align_corners=True)
            
            # Reshape for transformer
            feature = feature.flatten(2).transpose(1, 2)
            
            # Transform features
            output = transformer(feature)
            
            # Reshape back
            output = output.transpose(1, 2).reshape(B, C, H, W)
            outputs.append(output)
        
        # Combine scales with learned weights
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Resize all features to largest scale and combine
        target_size = outputs[0].shape[-2:]
        scaled_outputs = [
            F.interpolate(output, size=target_size, mode='bilinear', align_corners=True)
            for output in outputs
        ]
        
        return sum(w * out for w, out in zip(weights, scaled_outputs))
class EnhancedDetectionModel(nn.Module):
    def __init__(self, num_classes, num_queries=100):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = 256
        self.num_queries = num_queries
        
        # Backbones
        self.img_backbone = Backbone(input_channels=3)
        self.flow_backbone = Backbone(input_channels=2)
        
        # Feature Pyramid Network
        in_channels_list = [256, 512, 1024, 2048]  # ResNet50 output channels
        self.img_fpn = FeaturePyramidNetwork(in_channels_list, self.d_model)
        self.flow_fpn = FeaturePyramidNetwork(in_channels_list, self.d_model)
        
        # Feature fusion
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.d_model * 2, self.d_model, 1),
                nn.BatchNorm2d(self.d_model),
                nn.ReLU(inplace=True)
            )
            for _ in range(4)
        ])
        
        # Multi-scale transformer
        self.transformer = MultiScaleTransformer(self.d_model)
        
        # Query embeddings
        self.query_embed = nn.Parameter(torch.zeros(num_queries, self.d_model))
        self.query_feat = nn.Parameter(torch.zeros(num_queries, self.d_model))
        
        # Prediction heads
        self.bbox_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 4)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, num_classes + 1)
        )
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize query embeddings
        nn.init.xavier_uniform_(self.query_embed)
        nn.init.xavier_uniform_(self.query_feat)
        
        # Metrics
        self.map_metric = MeanAveragePrecision(
            box_format="cxcywh",
            iou_thresholds=[0.5],
            rec_thresholds=None,
            max_detection_thresholds=[1, 10, 100]
        )
        
    def forward(self, images, flows):
        # Extract multi-scale features
        img_features = self.img_backbone(images)
        flow_features = self.flow_backbone(flows)
        
        # Apply FPN
        img_features = self.img_fpn(img_features)
        flow_features = self.flow_fpn(flow_features)
        
        # Fuse features at each scale
        fused_features = [
            fusion(torch.cat([img_feat, flow_feat], dim=1))
            for fusion, img_feat, flow_feat in zip(self.fusion_layers, img_features, flow_features)
        ]
        
        # Process with multi-scale transformer
        transformer_output = self.transformer(fused_features)
        
        # Generate outputs
        B = images.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        query_feats = self.query_feat.unsqueeze(0).expand(B, -1, -1)
        
        # Predict boxes and classes
        boxes = self.bbox_head(query_feats).sigmoid()
        logits = self.class_head(query_feats)
        
        return boxes, logits
    
    def loss_fn(self, pred_boxes, pred_logits, target_boxes, target_labels):
        """Calculate loss with Hungarian matching"""
        B = pred_boxes.shape[0]
        device = pred_boxes.device
        
        # Loss weights
        box_loss_weight = 5.0
        cls_loss_weight = 2.0
        giou_loss_weight = 2.0
        
        total_loss = torch.tensor(0., device=device, requires_grad=True)
        num_boxes = 0
        
        for i in range(B):
            valid_mask = (target_labels[i] > 0) & (target_labels[i] < self.num_classes)
            batch_target_boxes = target_boxes[i][valid_mask]
            batch_target_labels = target_labels[i][valid_mask].long() - 1
            
            if valid_mask.sum() == 0:
                continue
                
            # Calculate cost matrix
            cost_bbox = torch.cdist(pred_boxes[i], batch_target_boxes, p=1)
            cost_giou = -self.generalized_box_iou(pred_boxes[i], batch_target_boxes)
            cost_class = -pred_logits[i][:, batch_target_labels]
            
            C = (cost_bbox * box_loss_weight + 
                 cost_class * cls_loss_weight + 
                 cost_giou * giou_loss_weight)
            
            indices = linear_sum_assignment(C.cpu().detach().numpy())
            indices = (torch.as_tensor(indices[0], dtype=torch.long), 
                      torch.as_tensor(indices[1], dtype=torch.long))
            
            num_boxes += len(batch_target_labels)
            
            # Classification loss
            target_classes = torch.full(pred_logits[i].shape[:1], self.num_classes,
                                      dtype=torch.int64, device=device)
            target_classes[indices[0]] = batch_target_labels[indices[1]]
            
            loss_ce = F.cross_entropy(pred_logits[i], target_classes)
            
            # Box losses
            matched_pred_boxes = pred_boxes[i][indices[0]]
            matched_target_boxes = batch_target_boxes[indices[1]]
            
            loss_bbox = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='none').sum(1)
            loss_giou = 1 - self.generalized_box_iou(matched_pred_boxes, matched_target_boxes)
            
            loss = (loss_bbox.mean() * box_loss_weight + 
                    loss_ce * cls_loss_weight + 
                    loss_giou.mean() * giou_loss_weight)
            
            total_loss = total_loss + loss
        
        return total_loss / max(num_boxes, 1)
    
    def generalized_box_iou(self, boxes1, boxes2):
        """
        Compute generalized IoU between two sets of boxes.
        Boxes are in cxcywh format.
        """
        # Convert to x1y1x2y2 format
        x1, y1, w1, h1 = boxes1.unbind(-1)
        x2, y2, w2, h2 = boxes2.unbind(-1)
        
        b1_x1 = x1 - w1/2
        b1_y1 = y1 - h1/2
        b1_x2 = x1 + w1/2
        b1_y2 = y1 + h1/2
        
        b2_x1 = x2 - w2/2
        b2_y1 = y2 - h2/2
        b2_x2 = x2 + w2/2
        b2_y2 = y2 + h2/2
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate intersection
        inter_x1 = torch.max(b1_x1.unsqueeze(-1), b2_x1)
        inter_y1 = torch.max(b1_y1.unsqueeze(-1), b2_y1)
        inter_x2 = torch.min(b1_x2.unsqueeze(-1), b2_x2)
        inter_y2 = torch.min(b1_y2.unsqueeze(-1), b2_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h
        
        # Calculate union
        union = area1.unsqueeze(-1) + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union.clamp(min=1e-6)
        
        # Calculate enclosing box
        encl_x1 = torch.min(b1_x1.unsqueeze(-1), b2_x1)
        encl_y1 = torch.min(b1_y1.unsqueeze(-1), b2_y1)
        encl_x2 = torch.max(b1_x2.unsqueeze(-1), b2_x2)
        encl_y2 = torch.max(b1_y2.unsqueeze(-1), b2_y2)
        
        encl_w = (encl_x2 - encl_x1).clamp(min=0)
        encl_h = (encl_y2 - encl_y1).clamp(min=0)
        enclosure = encl_w * encl_h
        
        # Calculate GIoU
        giou = iou - (enclosure - union) / enclosure.clamp(min=1e-6)
        
        return giou.squeeze(-1)
    
    def update_metrics(self, pred_boxes, pred_logits, targets):
        """Update evaluation metrics with proper post-processing"""
        predictions = []
        processed_targets = []
        
        # Confidence threshold
        conf_threshold = 0.5
        nms_threshold = 0.5
        
        for i in range(len(targets)):
            # Process predictions
            scores = torch.softmax(pred_logits[i], dim=-1)
            max_scores, pred_labels = scores.max(dim=-1)
            
            # Filter by confidence
            conf_mask = max_scores > conf_threshold
            filtered_boxes = pred_boxes[i][conf_mask]
            filtered_scores = max_scores[conf_mask]
            filtered_labels = pred_labels[conf_mask]
            
            # Convert boxes to xyxy format for NMS
            filtered_boxes_xyxy = box_cxcywh_to_xyxy(filtered_boxes)
            
            # Apply NMS per class
            final_boxes = []
            final_scores = []
            final_labels = []
            
            for class_id in filtered_labels.unique():
                class_mask = filtered_labels == class_id
                if not class_mask.any():
                    continue
                    
                class_boxes = filtered_boxes_xyxy[class_mask]
                class_scores = filtered_scores[class_mask]
                
                keep_indices = nms(class_boxes, class_scores, nms_threshold)
                
                final_boxes.append(filtered_boxes[class_mask][keep_indices])
                final_scores.append(filtered_scores[class_mask][keep_indices])
                final_labels.append(filtered_labels[class_mask][keep_indices])
            
            if final_boxes:
                predictions.append({
                    'boxes': torch.cat(final_boxes),
                    'scores': torch.cat(final_scores),
                    'labels': torch.cat(final_labels)
                })
            else:
                predictions.append({
                    'boxes': torch.zeros((0, 4), device=pred_boxes.device),
                    'scores': torch.zeros(0, device=pred_boxes.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=pred_boxes.device)
                })
            
            # Process targets
            valid_mask = targets[i]['labels'] > 0
            processed_targets.append({
                'boxes': targets[i]['boxes'][valid_mask],
                'labels': targets[i]['labels'][valid_mask]
            })
        
        self.map_metric.update(predictions, processed_targets)
    
    def get_metrics(self):
        """Get computed metrics"""
        metrics = self.map_metric.compute()
        return {
            'mAP50': metrics['map_50'].item(),
            'mAP50-95': metrics['map'].item(),
            'precision': metrics['map_per_class'].mean().item(),
            'recall': metrics['mar_100'].item(),
            'per_class_map': metrics['map_per_class'].tolist()
        }
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
