import torch
import yaml
from rep_extractors.conv_extractor import ConvExtractor
from leak_detector.sim_calculator import get_highest_similarity
from leak_detector.detector_head import LeakDetectorHead

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")

train_reps = torch.load('saved_results/conv_train_reps.pth').to(device)
test_reps = torch.load('saved_results/conv_train_reps.pth').to(device)   

highest_sim = get_highest_similarity(test_reps, train_reps).unsqueeze(1)
labels = torch.randint(0, 2, (len(highest_sim),))   

model = LeakDetectorHead().to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
loss_fn = torch.nn.CrossEntropyLoss()

for ep in range(cfg['training']['epochs']):
    opt.zero_grad()
    preds = model(highest_sim)
    loss = loss_fn(preds, labels)
    loss.backward()
    opt.step()
    print(f"Epoch {ep+1} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'saved_results/conv_detector.pth')
print("Conv based detector trained.")
