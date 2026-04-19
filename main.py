# Install dependencies
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

from torch.utils.data import Subset

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
        test.data[i] = train.data[idx]
        test.targets[i] = train.targets[idx]

    return train, test

train_dataset, test_dataset = create_leakage(train_dataset, test_dataset, 0.2)

# ✅ SMALL SUBSETS (FAST)
train_subset = Subset(train_dataset, list(range(2000)))
test_subset = Subset(test_dataset, list(range(500)))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

print("Data ready (FAST MODE)")

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*56*56,64)  # reduced size
        )

    def forward(self, x):
        return self.model(x)

cnn_model = CNNFeatureExtractor().to(device)


vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
vit_model.reset_classifier(0)
vit_model = vit_model.to(device)
vit_model.eval()

print("Tiny ViT ready")

def extract_features(model, loader):
    model.eval()
    features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            feats = model(images)
            features.append(feats.cpu().numpy())

    return np.vstack(features)

print("Extracting CNN features...")
cnn_train = extract_features(cnn_model, train_loader)
cnn_test = extract_features(cnn_model, test_loader)

print("Extracting ViT features...")
vit_train = extract_features(vit_model, train_loader)
vit_test = extract_features(vit_model, test_loader)

def compute_similarity(train_f, test_f):
    sim = cosine_similarity(test_f, train_f)
    return sim.max(axis=1)

cnn_sim = compute_similarity(cnn_train, cnn_test)
vit_sim = compute_similarity(vit_train, vit_test)

labels = np.zeros(len(test_subset))
labels[:int(0.2 * len(test_subset))] = 1

def evaluate(sim, labels, threshold=0.9):
    preds = (sim > threshold).astype(int)

    print("Accuracy:", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds))
    print("Recall:", recall_score(labels, preds))
    print("F1:", f1_score(labels, preds))

print("=== CNN RESULTS ===")
evaluate(cnn_sim, labels)

print("\n=== ViT RESULTS ===")
evaluate(vit_sim, labels)

import matplotlib.pyplot as plt

def plot_similarity(sim, labels, title):
    leaked = sim[labels == 1]
    clean = sim[labels == 0]

    plt.figure()

    plt.hist(
        leaked,
        bins=30,
        alpha=0.6,
        label="Leaked",
        color='red',
        edgecolor='black'
    )

    plt.hist(
        clean,
        bins=30,
        alpha=0.6,
        label="Clean",
        color='dodgerblue',
        edgecolor='black'
    )

    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

plot_similarity(cnn_sim, labels, "CNN Similarity Distribution")
plot_similarity(vit_sim, labels, "ViT Similarity Distribution")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion(sim, labels, threshold=0.9, title="Confusion Matrix"):
    preds = (sim > threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')  # strong contrast blue scale

    plt.title(title)
    plt.grid(False)
    plt.show()

plot_confusion(cnn_sim, labels, title="CNN Confusion Matrix")
plot_confusion(vit_sim, labels, title="ViT Confusion Matrix")

from sklearn.metrics import roc_curve, auc

def plot_roc(sim, labels, title):
    fpr, tpr, _ = roc_curve(labels, sim)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(
        fpr, tpr,
        color='red',
        linewidth=2,
        label=f"AUC = {roc_auc:.2f}"
    )

    plt.plot(
        [0, 1], [0, 1],
        color='black',
        linestyle='--'
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_roc(cnn_sim, labels, "CNN ROC Curve")
plot_roc(vit_sim, labels, "ViT ROC Curve")