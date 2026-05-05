import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)

def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(name)
    plt.close()

def plot_accuracy(history, name):
    plt.plot(history)
    plt.title("Accuracy")
    plt.savefig(name)
    plt.close()

def plot_comparison(results):
    names = list(results.keys())
    values = list(results.values())
    plt.bar(names, values)
    plt.savefig("model_comparison.png")
    plt.close()
