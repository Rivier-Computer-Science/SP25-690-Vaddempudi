import torch
import yaml
from torchvision import datasets, transforms
import numpy as np

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

tfm = transforms.ToTensor()
train_data = datasets.FashionMNIST(root=cfg['dataset']['root'], train=True, transform=tfm, download=True)
test_data = datasets.FashionMNIST(root=cfg['dataset']['root'], train=False, transform=tfm, download=True)

train_hashes = [hash(tuple(img.numpy().flatten())) for img, _ in train_data]
test_hashes = [hash(tuple(img.numpy().flatten())) for img, _ in test_data]

duplicate_count = sum(1 for h in test_hashes if h in train_hashes)
print(f"Simple hash baseline found {duplicate_count} exact matches in test set.")
