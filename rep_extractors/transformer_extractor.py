import torch
import torch.nn as nn

class TransformerExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3,64,4,4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64,128)

    def forward(self,x):
        x = self.patch(x)
        x = x.flatten(2).transpose(1,2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)
