import torch.nn as nn

class LeakDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )

    def forward(self,x):
        return self.net(x)
