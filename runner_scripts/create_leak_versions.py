import yaml
from data_builder.leak_creator import build_leak_split
import os

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

os.makedirs(cfg['dataset']['root'], exist_ok=True)
modes = ['clean', 'exact', 'near']
for m in modes:
    tr, te = build_leak_split(cfg['dataset']['root'], mode=m)
    print(f"Built {m} version - Train size: {len(tr)}, Test size: {len(te)}")
