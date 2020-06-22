# EXAMPLE_EVALUATE  Code to evaluate example results on ROxford and RParis datasets.
# Revisited protocol has 3 difficulty setups: Easy (E), Medium (M), and Hard (H), 
# and evaluates the performance using mean average precision (mAP), as well as mean precision @ k (mP@k)
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018
import argparse
import os

import h5py
import numpy as np

from revisitop_utils.dataset import configdataset
from revisitop_utils.download import download_datasets, download_features
from evaluate import compute_map

from sklearn.preprocessing import normalize
import faiss

# ---------------------------------------------------------------------
# Set data folder and testing parameters
# ---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
# Check, and, if necessary, download test data (Oxford and Pairs), 
# revisited annotation, and example feature vectors for evaluation
download_datasets(data_root)
download_features(data_root)

# Set test dataset: roxford5k | rparis6k
# test_dataset = 'roxford5k'


ap = argparse.ArgumentParser()

ap.add_argument("-p", "--paris", action='store_true',
                help="Test Paris6K")
ap.add_argument("-o", "--oxford", action='store_true',
                help="Test Oxford5K")

args = vars(ap.parse_args())

oxford_dataset = args['oxford']

test_dataset = 'roxford5k' if oxford_dataset else 'rparis6k'

# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

print('>> {}: Evaluating test dataset...'.format(test_dataset))
# config file for the dataset
# separates query image list from database image list, when revisited protocol used
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

# load query and database features
print('>> {}: Loading features...'.format(test_dataset))
# features = loadmat(os.path.join(data_root, 'features', '{}_resnet_rsfm120k_gem.mat'.format(test_dataset)))
# Q = features['Q']
# X = features['X']

# Q_dataset = h5py.File('Q_{}.h5'.format(test_dataset),'r')
# dim = Q_dataset[list(Q_dataset.keys())[0]][:].shape[0]
#
# n_queries = len(Q_dataset.keys())
# Q = np.zeros((n_queries, dim))
# for i in range(n_queries):
#     Q[i, :] = Q_dataset[str(i)][:]
#
# X_dataset = h5py.File('X_{}.h5'.format(test_dataset),'r')
# n_queries = len(X_dataset.keys())
# X = np.zeros((n_queries, dim))
# for i in range(n_queries):
#     X[i, :] = X_dataset[str(i)][:]

dataset_h5 = h5py.File('{}.h5'.format(test_dataset), 'r')

Q = dataset_h5['queries'][:].T
X = dataset_h5['database'][:].T

# perform search
print('>> {}: Retrieval...'.format(test_dataset))
sim_ = np.dot(X.T, Q)
ranks_ = np.argsort(-sim_, axis=0)


Q = Q.T.astype('float32')


X = X.T.astype('float32')

d = Q.shape[1]
distractors = np.random.random((int(300000), d)).astype('float32')
distractors = normalize(distractors)

nq = Q.shape[0]

X = np.vstack((X, distractors))
nb = X.shape[0]

print(X.shape)

assert Q.shape[1] == X.shape[1]

index = faiss.IndexFlatIP(d)   # build the index
nlist = 100
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(X)
assert index.is_trained

index.add(X)

index.nprobe = 1000

k = nb
sim, ranks = index.search(Q, k)
sim = np.array(sim).T
ranks = np.array(ranks).T

print(np.min(ranks))

# revisited evaluation
gnd = cfg['gnd']

# evaluate ranks
ks = [1, 5, 10]

# search for easy
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

# search for easy & hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

# search for hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
    gnd_t.append(g)
mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapE * 100, decimals=2),
                                              np.around(mapM * 100, decimals=2), np.around(mapH * 100, decimals=2)))
print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprE * 100, decimals=2),
                                                 np.around(mprM * 100, decimals=2), np.around(mprH * 100, decimals=2)))
