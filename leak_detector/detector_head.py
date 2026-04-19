import torch
import torch.nn as nn

class LeakDetectorHead(nn.Module):
    def __init__(self, input_size=1, hidden=96):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 2)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)
