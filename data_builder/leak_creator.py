import numpy as np

def create_leak_dataset(X_train, X_test, y_train, y_test, leak_ratio=0.2):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    n_leak = int(len(X_test) * leak_ratio)

    indices = np.random.choice(len(X_train), n_leak, replace=False)

    X_test[:n_leak] = X_train[indices]
    y_test[:n_leak] = y_train[indices]

    labels = np.zeros(len(X_test))
    labels[:n_leak] = 1

    return X_train, X_test, y_train, y_test, labels

def create_near_leak(X_train, X_test, y_train, y_test, leak_ratio=0.2):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    n_leak = int(len(X_test) * leak_ratio)

    indices = np.random.choice(len(X_train), n_leak, replace=False)

    noisy = X_train[indices] + np.random.normal(0, 0.1, X_train[indices].shape)
    noisy = np.clip(noisy, 0, 1)

    X_test[:n_leak] = noisy
    y_test[:n_leak] = y_train[indices]

    labels = np.zeros(len(X_test))
    labels[:n_leak] = 1

    return X_train, X_test, y_train, y_test, labels
