import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.detection import MeanAveragePrecision
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
        
    def forward(self, features):
        # Project concatenated features
        features = self.input_proj(features)  # [B, d_model, H, W]
        B, C, H, W = features.shape
        
        # Add positional encoding
        pos = self.pos_embed
        if H != pos.shape[2] or W != pos.shape[3]:
            pos = nn.functional.interpolate(pos, size=(H, W), mode='bicubic')
        features = features + pos
        
        # Reshape for transformer
        features = features.flatten(2).transpose(1, 2)  # [B, H*W, d_model]
        
        # Generate queries
        queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, d_model]
        
        # Transformer encoding and decoding
        memory = self.encoder(features)
        output = self.decoder(queries, memory)  # [B, num_queries, d_model]
        
        return output

class EnhancedDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = 256
        self.num_queries = 100
        
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
