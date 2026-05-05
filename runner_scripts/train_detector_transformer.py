import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from leak_detector.detector_head import LeakDetector

def train_detector(sim, labels):
    x = torch.tensor(sim).float().unsqueeze(1)
    y = torch.tensor(labels).long()
    ds = TensorDataset(x,y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = LeakDetector()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(10):
        correct = 0
        for xb,yb in dl:
            out = model(xb)
            loss = loss_fn(out,yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = out.argmax(1)
            correct += (pred==yb).sum().item()
        acc = correct/len(ds)
        history.append(acc)
    return model, history
