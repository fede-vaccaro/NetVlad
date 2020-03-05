import numpy as np


def transform(X, mean, components, explained_variance = None, pow_whiten=0.5, whiten=False):
    if mean is not None:
        X = X - mean
    X_transformed = np.dot(X, components.T)
    if whiten and explained_variance is not None:
        X_transformed /= np.power(explained_variance, pow_whiten)
    return X_transformed


def lr_warmup(it, frequency, min_lr=1e-6, max_lr=1e-5, step_factor=0.1, wu_steps=2000):
    if it < wu_steps:
        lr = max_lr * it / wu_steps + min_lr * (1. - it / wu_steps)
    else:
       n_cuts = int(np.floor(it / frequency))
       step_factor = step_factor ** min(n_cuts, 1)
       lr = max_lr*step_factor

    return lr

