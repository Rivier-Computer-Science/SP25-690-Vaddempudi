!pip install timm scikit-learn

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

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

# Fast subsets
train_subset = Subset(train_dataset, list(range(2000)))
test_subset = Subset(test_dataset, list(range(500)))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

cnn_model = timm.create_model('resnet18', pretrained=True, num_classes=10).to(device)
vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10).to(device)


def train_model(model, loader, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("Training CNN...")
train_model(cnn_model, train_loader, epochs=2)

print("Training ViT...")
train_model(vit_model, train_loader, epochs=2)


cnn_model.reset_classifier(0)
vit_model.reset_classifier(0)

cnn_model.eval()
vit_model.eval()

def extract_features(model, loader):
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            f = model(x)
            feats.append(f.cpu().numpy())
    return np.vstack(feats)

cnn_train = extract_features(cnn_model, train_loader)
cnn_test = extract_features(cnn_model, test_loader)

vit_train = extract_features(vit_model, train_loader)
vit_test = extract_features(vit_model, test_loader)



def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

cnn_train = normalize(cnn_train)
cnn_test = normalize(cnn_test)

vit_train = normalize(vit_train)
vit_test = normalize(vit_test)


def compute_similarity(train_f, test_f):
    sim = cosine_similarity(test_f, train_f)
    return sim.max(axis=1)

cnn_sim = compute_similarity(cnn_train, cnn_test)
vit_sim = compute_similarity(vit_train, vit_test)


labels = np.zeros(len(test_subset))
labels[:int(0.2 * len(test_subset))] = 1

def find_best_threshold(sim, labels):
    best_t, best_f1 = 0, 0
    for t in np.linspace(0.5, 0.99, 50):
        preds = (sim > t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t

cnn_thresh = find_best_threshold(cnn_sim, labels)
vit_thresh = find_best_threshold(vit_sim, labels)


def evaluate(sim, labels, threshold):
    preds = (sim > threshold).astype(int)

    print("Accuracy:", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds))
    print("Recall:", recall_score(labels, preds))
    print("F1:", f1_score(labels, preds))

print("=== CNN RESULTS ===")
evaluate(cnn_sim, labels, cnn_thresh)

print("\n=== ViT RESULTS ===")
evaluate(vit_sim, labels, vit_thresh)


plt.hist(vit_sim[labels==1], bins=30, alpha=0.6, color='red', label='Leaked', edgecolor='black')
plt.hist(vit_sim[labels==0], bins=30, alpha=0.6, color='blue', label='Clean', edgecolor='black')
plt.legend()
plt.title("Similarity Distribution")
plt.show()


fpr, tpr, _ = roc_curve(labels, vit_sim)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='red', label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--', color='black')
plt.legend()
plt.title("ROC Curve")
plt.show()


preds = (vit_sim > vit_thresh).astype(int)
cm = confusion_matrix(labels, preds)

ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

