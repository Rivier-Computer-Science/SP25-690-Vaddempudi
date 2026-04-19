import torch
from torchvision import datasets, transforms
import numpy as np

def build_leak_split(root, mode='clean', download=True):
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root=root, train=True, transform=transform, download=download)
    test_data = datasets.FashionMNIST(root=root, train=False, transform=transform, download=download)
    
    if mode == 'clean':
        return train_data, test_data
    
    train_imgs = [img for img, _ in train_data]
    train_lbls = [lbl for _, lbl in train_data]
    
    if mode == 'exact':
        leak_count = int(len(train_imgs) * 0.08)
        leak_ids = np.random.choice(len(train_imgs), leak_count, replace=False)
        for i in leak_ids:
            test_data.data = np.vstack((test_data.data, np.array(train_imgs[i])))
            test_data.targets = np.append(test_data.targets, train_lbls[i])
    
    elif mode == 'near':
        leak_count = int(len(train_imgs) * 0.08)
        leak_ids = np.random.choice(len(train_imgs), leak_count, replace=False)
        for i in leak_ids:
            img = train_imgs[i]
            if np.random.rand() < 0.6:
                img = transforms.RandomHorizontalFlip()(img)
            if np.random.rand() < 0.5:
                img = transforms.RandomRotation(12)(img)
            test_data.data = np.vstack((test_data.data, np.array(img)))
            test_data.targets = np.append(test_data.targets, train_lbls[i])
    
    return train_data, test_data
