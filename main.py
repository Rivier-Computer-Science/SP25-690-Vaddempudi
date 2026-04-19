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
print(device)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

def create_leakage(train, test, ratio=0.2):
    num = int(len(test)*ratio)
    idxs = random.sample(range(len(train)), num)

    for i, idx in enumerate(idxs):
        test.data[i] = train.data[idx]
        test.targets[i] = train.targets[idx]

    return train, test

train_dataset, test_dataset = create_leakage(train_dataset, test_dataset)

train_subset = Subset(train_dataset, list(range(2000)))
test_subset = Subset(test_dataset, list(range(500)))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

cnn_model = timm.create_model("resnet18", pretrained=True)
cnn_model.fc = nn.Identity()   # 🔥 key fix
cnn_model = cnn_model.to(device)

vit_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
vit_model.reset_classifier(0)  # feature mode
vit_model = vit_model.to(device)
def train(model, loader, epochs=2):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epochs):
        total = 0
        for x,y in loader:
            x,y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()

            total += loss.item()

        print("Epoch", e+1, "Loss:", total)

print("CNN Training")
train(cnn_model, train_loader)

print("ViT Training")
train(vit_model, train_loader)
def extract(model, loader):
    model.eval()
    feats = []

    with torch.no_grad():
        for x,_ in loader:
            x = x.to(device)

            if "vit" in str(type(model)):
                f = model.forward_features(x)   # 🔥 FIX
                f = f[:,0]  # CLS token
            else:
                f = model(x)

            feats.append(f.cpu().numpy())

    return np.vstack(feats)

cnn_train = extract(cnn_model, train_loader)
cnn_test = extract(cnn_model, test_loader)

vit_train = extract(vit_model, train_loader)
vit_test = extract(vit_model, test_loader)

def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

cnn_train = norm(cnn_train)
cnn_test = norm(cnn_test)

vit_train = norm(vit_train)
vit_test = norm(vit_test)

def similarity(train_f, test_f):
    return cosine_similarity(test_f, train_f).max(axis=1)

cnn_sim = similarity(cnn_train, cnn_test)
vit_sim = similarity(vit_train, vit_test)

labels = np.zeros(len(test_subset))
labels[:int(0.2*len(test_subset))] = 1

def best_threshold(sim, labels):
    best_t, best_f1 = 0, 0

    for t in np.linspace(0.5,0.99,50):
        pred = (sim>t).astype(int)
        f1 = f1_score(labels,pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t

cnn_t = best_threshold(cnn_sim, labels)
vit_t = best_threshold(vit_sim, labels)

def evaluate(sim, labels, t):
    pred = (sim>t).astype(int)

    print("Accuracy:", accuracy_score(labels,pred))
    print("Precision:", precision_score(labels,pred))
    print("Recall:", recall_score(labels,pred))
    print("F1:", f1_score(labels,pred))

print("CNN")
evaluate(cnn_sim, labels, cnn_t)

print("\nViT")
evaluate(vit_sim, labels, vit_t)

fpr, tpr, _ = roc_curve(labels, vit_sim)
auc_score = auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"AUC={auc_score:.2f}")
plt.plot([0,1],[0,1],"--")
plt.legend()
plt.title("ROC Curve")
plt.show()
