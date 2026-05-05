import torch

def extract(model, loader):
    feats = []
    model.eval()
    with torch.no_grad():
        for x,_ in loader:
            feats.append(model(x).cpu())
    return torch.cat(feats).numpy()
