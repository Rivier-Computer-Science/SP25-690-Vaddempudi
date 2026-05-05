import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import *
from data_builder.leak_creator import *
from rep_extractors.conv_extractor import ConvExtractor
from rep_extractors.transformer_extractor import TransformerExtractor
from leak_detector.sim_calculator import compute_similarity
from runner_scripts.train_detector_conv import train_detector
from runner_scripts.simple_hash_baseline import hash_baseline
from runner_scripts.pull_representations import extract

set_seed()

transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

X_train = train_data.data.astype('float32')/255
X_test = test_data.data.astype('float32')/255
y_train = train_data.targets
y_test = test_data.targets

X_train, X_test, y_train, y_test, labels = create_leak_dataset(X_train, X_test, y_train, y_test)

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

conv = ConvExtractor()
trans = TransformerExtractor()

train_feats_conv = extract(conv, train_loader)
test_feats_conv = extract(conv, test_loader)

train_feats_trans = extract(trans, train_loader)
test_feats_trans = extract(trans, test_loader)

sim_conv = compute_similarity(train_feats_conv, test_feats_conv)
sim_trans = compute_similarity(train_feats_trans, test_feats_trans)

model_conv, hist_conv = train_detector(sim_conv, labels)
model_trans, hist_trans = train_detector(sim_trans, labels)

pred_conv = model_conv(torch.tensor(sim_conv).float().unsqueeze(1)).argmax(1).numpy()
pred_trans = model_trans(torch.tensor(sim_trans).float().unsqueeze(1)).argmax(1).numpy()
pred_hash = hash_baseline(X_train, X_test)

plot_confusion(labels, pred_conv, "confusion_conv.png")
plot_confusion(labels, pred_trans, "confusion_transformer.png")

plot_accuracy(hist_conv, "conv_accuracy.png")
plot_accuracy(hist_trans, "transformer_accuracy.png")

results = {
    "hash": (pred_hash==labels).mean(),
    "conv": (pred_conv==labels).mean(),
    "transformer": (pred_trans==labels).mean()
}

plot_comparison(results)

print(results)
