# Install dependencies
!pip install torch torchvision timm scikit-learn matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

def create_leakage(train, test, ratio=0.2):
    num_leak = int(len(test) * ratio)
    indices = random.sample(range(len(train)), num_leak)

    for i, idx in enumerate(indices):
        # Copy RAW image (32x32) directly
        test.data[i] = train.data[idx]
        test.targets[i] = train.targets[idx]

    return train, test

train_dataset, test_dataset = create_leakage(train_dataset, test_dataset, 0.2)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Data ready with leakage")
