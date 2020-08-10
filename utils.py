import numpy as np
import time
import torch

def predict_with_netvlad(img_tensor, model, device, output_dim=32768, batch_size=16,verbose=False):
    n_imgs = img_tensor.shape[0]
    descs = np.zeros((n_imgs, output_dim))
    n_iters = int(np.ceil(n_imgs / batch_size))

    with torch.no_grad():
        model.eval()
        if verbose:
            print("")
        for i in range(n_iters):
            low = i * batch_size
            high = int(np.min([n_imgs, (i + 1) * batch_size]))
            batch_gpu = img_tensor[low:high].to(device)
            out_batch = model.forward(batch_gpu).cpu().numpy()
            descs[low:high] = out_batch
            if verbose:
                print("\r>> Predicted batch {}/{}".format(i + 1, n_iters), end='')
        if verbose:
            print("")

    return descs


def predict_generator_with_netlvad(model, device, generator, n_steps, verbose=True):
    descs = []

    t0 = time.time()
    with torch.no_grad():
        model.eval()
        print("")
        for i, X in enumerate(generator):
            if type(X) is type(tuple()) or type(X) is type(list()):
                x = X[0]
            else:
                x = X
            batch_gpu = x.to(device)
            out_batch = model.forward(batch_gpu).cpu().numpy()
            descs.append(out_batch)
            if verbose:
                print("\r>> Predicted (w/ generator) batch {}/{}".format(i + 1, n_steps), end='')
            if i + 1 == n_steps:
                break
        if verbose:
            print("\n>> Prediction completed in {}s".format(int(time.time() - t0)))

    descs = np.vstack([m for m in descs])
    return descs


def transform(X, mean, components, explained_variance=None, pow_whiten=0.5, whiten=False):
    if mean is not None:
        X = X - mean
    X_transformed = np.dot(X, components.T)
    if whiten and explained_variance is not None:
        X_transformed /= np.power(explained_variance, pow_whiten)
    return X_transformed


def lr_warmup(frequency, min_lr=1e-6, max_lr=1e-5, step_factor=0.1, wu_steps=2000, weight_decay=2e-6):
    def funct(it, min_lr, max_lr, step_factor, wu_steps):
        if it < wu_steps:
            lr = max_lr * it / wu_steps + min_lr * (1. - it / wu_steps)
        else:
            n_cuts = int(np.floor(it / frequency))
            step_factor = step_factor ** n_cuts
            lr = max_lr * step_factor

        return lr  * 1 / (1 + (it % frequency) * weight_decay)

    lambda_lr = lambda it: funct(it, min_lr=min_lr, max_lr=max_lr, step_factor=step_factor, wu_steps=wu_steps)

    return lambda_lr
