import numpy as np
from utils import cosine_similarity

def compute_similarity(train_feats, test_feats):
    sims = []
    for t in test_feats:
        sim = cosine_similarity(train_feats, t.reshape(1,-1))
        sims.append(np.max(sim))
    return np.array(sims)
