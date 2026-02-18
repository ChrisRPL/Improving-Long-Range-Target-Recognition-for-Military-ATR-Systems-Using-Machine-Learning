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

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, size):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, d_model, size, size))
        nn.init.normal_(self.embedding, mean=0, std=0.02)
    
    def forward(self, x):
        return self.embedding

class MultiScaleTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3):  # Reduced num_layers
        super().__init__()
        self.d_model = d_model
        
        # Reduce number of scales
        self.scales = [32, 16, 8]  # Removed largest scale to save memory
        
        # Create position embedding modules properly
        self.pos_embeddings = nn.ModuleList([
            PositionalEmbedding(d_model, size)
            for size in self.scales
        ])
        
        # Scale-specific transformers with reduced complexity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,  # Reduced feedforward dimension
            batch_first=True,
            dropout=0.1,  # Added dropout for regularization
        )
        
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers)
            for _ in range(len(self.scales))
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))
        
        # Add memory-efficient fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model * len(self.scales), d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        # Process only selected scales
        outputs = []
        
        for idx, (feature, pos_embed, transformer) in enumerate(zip(
            features[-len(self.scales):],  # Take only the scales we want
            self.pos_embeddings,
            self.transformers
        )):
            B, C, H, W = feature.shape
            
            # Add positional encoding
            pos_embedding = pos_embed(feature)
            feature = feature + F.interpolate(pos_embedding, size=(H, W), mode='bilinear', align_corners=True)
            
            # Process in chunks if feature is too large
            if H * W > 1024:  # Threshold for chunking
                chunks = []
                chunk_size = 1024
                for i in range(0, H * W, chunk_size):
                    # Reshape and chunk
                    chunk = feature.flatten(2)[:, :, i:i+chunk_size]
                    if chunk.size(2) == 0:
                        continue
                    
                    chunk = chunk.transpose(1, 2)
                    chunk = transformer(chunk)
                    chunk = chunk.transpose(1, 2)
                    chunks.append(chunk)
                
                # Combine chunks
                feature = torch.cat(chunks, dim=-1)
                feature = feature.reshape(B, C, H, W)
            else:
                # Process normally for small features
                feature = feature.flatten(2).transpose(1, 2)
                feature = transformer(feature)
                feature = feature.transpose(1, 2).reshape(B, C, H, W)
            
            outputs.append(feature)
        
        # Memory efficient fusion
        target_size = outputs[0].shape[-2:]
        scaled_outputs = []
        
        for output, weight in zip(outputs, F.softmax(self.scale_weights, dim=0)):
            scaled_output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=True)
            scaled_outputs.append(scaled_output * weight)
        
        # Concatenate and fuse
        multi_scale_features = torch.cat(scaled_outputs, dim=1)
        fused_features = self.fusion(multi_scale_features)
        
        # Clean up to free memory
        del outputs, scaled_outputs, multi_scale_features
        torch.cuda.empty_cache()
        
        return fused_features
        
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
        """Calculate loss with proper box and class handling"""
        B = pred_boxes.shape[0]
        device = pred_boxes.device
    
        # Loss weights
        box_loss_weight = 5.0
        cls_loss_weight = 2.0
        giou_loss_weight = 2.0
    
        total_loss = torch.tensor(0., device=device, requires_grad=True)
        num_boxes = 0
    
        for i in range(B):
            # Get valid targets for this batch
            valid_mask = target_labels[i] > 0  # Keep all valid classes
            batch_target_boxes = target_boxes[i][valid_mask]
            batch_target_labels = target_labels[i][valid_mask].long()
        
            # Handle empty targets
            if valid_mask.sum() == 0:
                # Background classification loss
                bg_loss = F.cross_entropy(
                    pred_logits[i],
                    torch.zeros(pred_logits.shape[1], dtype=torch.long, device=device)
                )
                total_loss = total_loss + bg_loss * 0.1
                continue
        
            # Calculate cost matrix
            cost_bbox = torch.cdist(pred_boxes[i], batch_target_boxes, p=1)
        
            # Classification cost - handle background class properly
            class_probs = F.softmax(pred_logits[i], dim=-1)
            cost_class = -class_probs[:, batch_target_labels]
        
            # GIoU cost
            with torch.no_grad():
                cost_giou = -self.generalized_box_iou(
                    pred_boxes[i],
                    batch_target_boxes
                )
        
            # Combine costs
            C = (
            cost_bbox * box_loss_weight +
            cost_class * cls_loss_weight +
            cost_giou * giou_loss_weight
            )
        
            # Hungarian matching with cost matrix on CPU
            indices = linear_sum_assignment(C.detach().cpu().numpy())
            indices = (
                torch.as_tensor(indices[0], dtype=torch.long, device=device),
                torch.as_tensor(indices[1], dtype=torch.long, device=device)
            )
        
            num_boxes += len(batch_target_labels)
        
            # Classification loss for all queries
            target_classes = torch.full(
                pred_logits[i].shape[:1], 
                0,  # Background class
                dtype=torch.long, 
                device=device
            )
            target_classes[indices[0]] = batch_target_labels[indices[1]]
        
            loss_ce = F.cross_entropy(
                pred_logits[i],
                target_classes,
                label_smoothing=0.1  # Add label smoothing
            )
        
            # Box losses for matched pairs
            if len(indices[0]) > 0:
                matched_pred_boxes = pred_boxes[i][indices[0]]
                matched_target_boxes = batch_target_boxes[indices[1]]
            
                # L1 loss
                loss_bbox = F.l1_loss(
                matched_pred_boxes, 
                matched_target_boxes, 
                reduction='none'
                ).sum(1).mean()
            
                # GIoU loss
                loss_giou = (1 - self.generalized_box_iou(
                matched_pred_boxes,
                matched_target_boxes
                )).mean()
            
                # Combine box losses
                box_loss = loss_bbox * box_loss_weight + loss_giou * giou_loss_weight
            else:
                box_loss = torch.tensor(0., device=device)
        
            # Final loss
            batch_loss = box_loss + loss_ce * cls_loss_weight
            total_loss = total_loss + batch_loss
    
        return total_loss / max(num_boxes, 1)
    
    def generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute generalized IoU between two sets of boxes.
        Both sets of boxes are expected to be in (x_center, y_center, width, height) format.
    
        Args:
            boxes1: (N, 4) tensor containing first set of boxes
            boxes2: (M, 4) tensor containing second set of boxes
        
        Returns:
            giou: (N, M) tensor containing pairwise GIoU values
        """
        # Make sure inputs are valid
        if not (boxes1.size(-1) == boxes2.size(-1) == 4):
            raise ValueError("Both box tensors must have shape (..., 4)")
    
        # Add singleton dimension for broadcasting if needed
        if boxes1.dim() == 1:
            boxes1 = boxes1.unsqueeze(0)
        if boxes2.dim() == 1:
            boxes2 = boxes2.unsqueeze(0)
    
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        x1_1 = boxes1[..., 0] - boxes1[..., 2] / 2
        y1_1 = boxes1[..., 1] - boxes1[..., 3] / 2
        x2_1 = boxes1[..., 0] + boxes1[..., 2] / 2
        y2_1 = boxes1[..., 1] + boxes1[..., 3] / 2
    
        x1_2 = boxes2[..., 0] - boxes2[..., 2] / 2
        y1_2 = boxes2[..., 1] - boxes2[..., 3] / 2
        x2_2 = boxes2[..., 0] + boxes2[..., 2] / 2
        y2_2 = boxes2[..., 1] + boxes2[..., 3] / 2
    
        # Calculate areas
        area1 = boxes1[..., 2] * boxes1[..., 3]
        area2 = boxes2[..., 2] * boxes2[..., 3]
    
        # Calculate intersection coordinates
        x1_i = torch.max(x1_1.unsqueeze(-1), x1_2)
        y1_i = torch.max(y1_1.unsqueeze(-1), y1_2)
        x2_i = torch.min(x2_1.unsqueeze(-1), x2_2)
        y2_i = torch.min(y2_1.unsqueeze(-1), y2_2)
    
        # Calculate intersection area
        w_i = (x2_i - x1_i).clamp(min=0)
        h_i = (y2_i - y1_i).clamp(min=0)
        intersection = w_i * h_i
    
        # Calculate union area
        union = area1.unsqueeze(-1) + area2 - intersection
    
        # Calculate IoU
        iou = intersection / union.clamp(min=1e-6)
    
        # Calculate enclosing box coordinates
        x1_c = torch.min(x1_1.unsqueeze(-1), x1_2)
        y1_c = torch.min(y1_1.unsqueeze(-1), y1_2)
        x2_c = torch.max(x2_1.unsqueeze(-1), x2_2)
        y2_c = torch.max(y2_1.unsqueeze(-1), y2_2)
    
        # Calculate enclosing box area
        w_c = (x2_c - x1_c).clamp(min=0)
        h_c = (y2_c - y1_c).clamp(min=0)
        enclosure = w_c * h_c
    
        # Calculate GIoU
        giou = iou - (enclosure - union) / enclosure.clamp(min=1e-6)
    
        return giou
        
	def update_metrics(self, pred_boxes, pred_logits, target_boxes, target_labels):
		"""Update evaluation metrics with proper post-processing"""
		predictions = []
		targets = []
		
		batch_size = pred_boxes.size(0)
		
		for i in range(batch_size):
		    # Get valid targets
		    valid_mask = target_labels[i] > 0
		    batch_target_boxes = target_boxes[i][valid_mask]
		    batch_target_labels = target_labels[i][valid_mask]
		    
		    # Process predictions
		    scores = F.softmax(pred_logits[i], dim=-1)
		    max_scores, pred_labels = scores.max(dim=-1)
		    
		    # Filter background predictions
		    fg_mask = pred_labels > 0
		    filtered_boxes = pred_boxes[i][fg_mask]
		    filtered_scores = max_scores[fg_mask]
		    filtered_labels = pred_labels[fg_mask]
		    
		    # Apply NMS per class
		    if len(filtered_boxes) > 0:
		        # Convert boxes to xyxy for NMS
		        boxes_xyxy = self.box_cxcywh_to_xyxy(filtered_boxes)
		        
		        # Apply NMS per class
		        keep_indices = []
		        for class_id in filtered_labels.unique():
		            class_mask = filtered_labels == class_id
		            if not class_mask.any():
		                continue
		            
		            class_boxes = boxes_xyxy[class_mask]
		            class_scores = filtered_scores[class_mask]
		            
		            class_keep = nms(class_boxes, class_scores, iou_threshold=0.5)
		            class_indices = torch.where(class_mask)[0][class_keep]
		            keep_indices.extend(class_indices.tolist())
		        
		        keep_indices = torch.tensor(keep_indices, device=filtered_boxes.device)
		        
		        # Update predictions
		        predictions.append({
		            'boxes': filtered_boxes[keep_indices],
		            'scores': filtered_scores[keep_indices],
		            'labels': filtered_labels[keep_indices]
		        })
		    else:
		        predictions.append({
		            'boxes': torch.zeros((0, 4), device=pred_boxes.device),
		            'scores': torch.zeros(0, device=pred_boxes.device),
		            'labels': torch.zeros(0, dtype=torch.long, device=pred_boxes.device)
		        })
		    
		    # Update targets
		    if len(batch_target_boxes) > 0:
		        targets.append({
		            'boxes': batch_target_boxes,
		            'labels': batch_target_labels
		        })
		    else:
		        targets.append({
		            'boxes': torch.zeros((0, 4), device=target_boxes.device),
		            'labels': torch.zeros(0, dtype=torch.long, device=target_boxes.device)
		        })
		
		# Update metrics
		self.map_metric.update(predictions, targets)

	def get_metrics(self):
		"""Get computed metrics with proper handling"""
		metrics = self.map_metric.compute()
		
		# Handle empty metrics case
		if metrics['map_50'] is None or metrics['map'] is None:
		    return {
		        'mAP50': 0.0,
		        'mAP50-95': 0.0,
		        'precision': 0.0,
		        'recall': 0.0,
		        'per_class_map': [0.0] * self.num_classes
		    }
		
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
