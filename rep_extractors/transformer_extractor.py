import torch
import torch.nn as nn
import timm

class TransformerExtractor(nn.Module):
    def __init__(self, feat_size=384):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        self.linear = nn.Linear(self.backbone.num_features, feat_size)
    
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        cls_token = feats[:, 0]
        return self.linear(cls_token)
