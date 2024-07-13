import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import box_iou
import torch.nn.functional as F
import math

# Backbone Network (ResNet)
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove fully connected layer

    def forward(self, x):
        return self.resnet(x)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer Encoder-Decoder
class TransformerModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=100):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead), num_decoder_layers)
        self.object_queries = nn.Parameter(torch.rand(num_queries, d_model))
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, src):
        src = self.positional_encoding(src)
        memory = self.encoder(src)
        tgt = self.object_queries.unsqueeze(1).repeat(1, src.size(1), 1)  # (num_queries, batch_size, d_model)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, memory)
        return output

# Prediction Heads
class BBoxHead(nn.Module):
    def __init__(self, d_model):
        super(BBoxHead, self).__init__()
        self.linear = nn.Linear(d_model, 4)

    def forward(self, x):
        return self.linear(x)

class ClassHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ClassHead, self).__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.linear(x)

# Full Model
class DetectionModel(nn.Module):
    def __init__(self, num_classes, backbone, transformer, bbox_head, class_head):
        super(DetectionModel, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.bbox_head = bbox_head
        self.class_head = class_head

    def forward(self, images, flows):
        # Feature extraction
        features_image = self.backbone(images)
        features_flow = self.backbone(flows)
        features = torch.cat([features_image, features_flow], dim=1)

        # Flatten spatial dimensions and permute for transformer
        b, c, h, w = features.size()
        features = features.view(b, c, -1).permute(2, 0, 1)

        # Transformer
        transformer_output = self.transformer(features)

        # Prediction heads
        bboxes = self.bbox_head(transformer_output)
        class_logits = self.class_head(transformer_output)

        return bboxes, class_logits

def create_model(num_classes):
    backbone = Backbone()
    transformer = TransformerModel()
    bbox_head = BBoxHead(d_model=512)
    class_head = ClassHead(d_model=512, num_classes=num_classes)
    return DetectionModel(num_classes=num_classes, backbone=backbone, transformer=transformer, bbox_head=bbox_head, class_head=class_head)

