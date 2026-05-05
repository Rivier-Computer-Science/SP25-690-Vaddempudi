import numpy as np

def hash_baseline(X_train, X_test):
    hashes = set([x.tobytes() for x in X_train])
    preds = []
    for x in X_test:
        preds.append(1 if x.tobytes() in hashes else 0)
    return np.array(preds)
