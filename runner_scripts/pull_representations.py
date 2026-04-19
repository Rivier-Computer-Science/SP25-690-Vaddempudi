import torch
import yaml
from rep_extractors.conv_extractor import ConvExtractor
from rep_extractors.transformer_extractor import TransformerExtractor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")

conv_net = ConvExtractor().to(device)
trans_net = TransformerExtractor().to(device)

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
train_set = datasets.FashionMNIST(root=cfg['dataset']['root'], train=True, transform=tfm, download=True)
loader = DataLoader(train_set, batch_size=cfg['dataset']['batch_size'], shuffle=False)

conv_feats = []
trans_feats = []
for imgs, _ in loader:
    imgs = imgs.to(device)
    with torch.no_grad():
        conv_feats.append(conv_net(imgs))
        trans_feats.append(trans_net(imgs))

conv_feats = torch.cat(conv_feats)
trans_feats = torch.cat(trans_feats)

torch.save(conv_feats, 'saved_results/conv_train_reps.pth')
torch.save(trans_feats, 'saved_results/transformer_train_reps.pth')
print("Representations extracted and saved.")
