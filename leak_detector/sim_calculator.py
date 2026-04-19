import torch
import torch.nn.functional as F

def get_highest_similarity(test_reps, train_reps):
    sims = F.cosine_similarity(test_reps.unsqueeze(1), train_reps.unsqueeze(0), dim=2)
    highest, _ = sims.max(dim=1)
    return highest
