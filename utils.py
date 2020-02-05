import numpy as np


def transform(X, mean, components, explained_variance = None, pow_whiten=0.5, whiten=False):
    if mean is not None:
        X = X - mean
    X_transformed = np.dot(X, components.T)
    if whiten and explained_variance is not None:
        X_transformed /= np.power(explained_variance, pow_whiten)
    return X_transformed


def lr_warmup(it, min_lr=1e-6, max_lr=1e-5, wu_steps=2000):
    if it < wu_steps:
        lr = max_lr * it / wu_steps + min_lr * (1. - it / wu_steps)
    else:
        lr = max_lr

    return lr
