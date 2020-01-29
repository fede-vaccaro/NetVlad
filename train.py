# %%
import argparse
import gc
import os
import time
from math import ceil

import clr_callback as clr
import h5py
import holidays_testing_helpers as hth
import matplotlib.pyplot as plt
import numpy as np
import open_dataset_utils as my_utils
import paths
from keras import Model, optimizers
from keras import backend as K
from keras import layers, regularizers
from netvlad_model import NetVLADSiameseModel, input_shape, NetVladResnet
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
# from keras_radam import RAdam
# from keras_radam.training import RAdamOptimizer
from tqdm import tqdm
from triplet_loss import TripletLossLayer
import tensorflow as tf
import keras
import yaml

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
ap.add_argument("-c", "--configuration", type=str, default='train_configuration.yaml',
                help="Yaml file where the configuration is stored")
ap.add_argument("-t", "--test", action='store_true',
                help="If the training be bypassed for testing")
ap.add_argument("-k", "--kmeans", action='store_true',
                help="If netvlad weights should be initialized for testing")
ap.add_argument("-d", "--device", type=str, default="0",
                help="CUDA device to be used. For info type '$ nvidia-smi'")
args = vars(ap.parse_args())

test = args['test']
test_kmeans = args['kmeans']
model_name = args['model']
cuda_device = args['device']
config_file = args['configuration']

def lr_warmup(it, min_lr=1e-6, max_lr=1e-5, wu_steps=2000):
    if it < wu_steps:
        lr = max_lr * it / wu_steps + min_lr * (1. - it / wu_steps)
    else:
        lr = max_lr

    return lr


conf_file = open(config_file, 'r')
conf = dict(yaml.safe_load(conf_file))
conf_file.close()

network_conf = conf['network']
net_name = network_conf['name']

n_cluster = network_conf['n_clusters']
middle_pca = network_conf['middle_pca']

train_description = conf['description']
mining_batch_size = conf['mining_batch_size']
minibatch_size = conf['minibatch_size']
steps_per_epoch = conf['steps_per_epoch']
epochs = conf['n_epochs']

lr_conf = conf['lr']
use_warm_up = lr_conf['warm-up']
warm_up_steps = lr_conf['warm-up-steps']
max_lr = float(lr_conf['max_value'])

rotate_holidays = conf['rotate_holidays']
use_power_norm = conf['use_power_norm']
use_multi_resolution = conf['use_multi_resolution']

# if test:
#     gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#     for device in gpu_devices:
#         tf.config.experimental.set_memory_growth(device, True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

my_model = None
if net_name == "vgg":
    my_model = NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    my_model = NetVladResnet(**network_conf)
else:
    print("Network name not valid.")

index, classes = my_utils.generate_index_mirflickr(paths.mirflickr_annotations)
files = [paths.mirflickr_path + k for k in list(index.keys())]

vgg, output_shape = my_model.get_feature_extractor(verbose=True)

generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=160)
vgg_netvlad = my_model.build_netvladmodel()
vgg_netvlad.summary()

print("Netvlad output shape: ", vgg_netvlad.output_shape)
print("Feature extractor output shape: ", vgg.output_shape)

train_pca = False
train_kmeans = (not test or test_kmeans) and model_name is None and not train_pca
train = not test

if train_kmeans:
    print("Predicting local features for k-means. Output shape: ", output_shape)
    all_descs = vgg.predict_generator(generator=generator_nolabels, steps=10, verbose=1)
    print("All descs shape: ", all_descs.shape)

    locals = np.vstack((m[np.random.randint(len(m), size=150)] for m in all_descs)).astype('float32')

    print("Sampling local features")

    locals = normalize(locals, axis=1)
    np.random.shuffle(locals)

    if middle_pca['pretrain'] and middle_pca['active']:
        print("Training PCA")
        pca = PCA(middle_pca['dim'])
        locals = pca.fit_transform(locals)
        my_model.set_mid_pca_weights(pca)
    print("Locals extracted: {}".format(locals.shape))

    n_clust = my_model.n_cluster
    print("Fitting k-means")
    kmeans = MiniBatchKMeans(n_clusters=n_clust).fit(locals[locals.shape[0] // 3:])

    print("Initializing NetVLAD")
    my_model.set_netvlad_weights(kmeans)

    del all_descs
    gc.collect()

if train:
    steps_per_epoch = steps_per_epoch

    vgg_netvlad.summary()

    start_epoch = int(args['start_epoch'])
    vgg_netvlad = Model(vgg_netvlad.input, TripletLossLayer(0.1)(vgg_netvlad.output))

    lr = 1e-5
    cyclic = clr.CyclicLR(base_lr=1e-6, max_lr=1e-5, mode='exp_range', gamma=0.99993)

    if model_name is not None:
        print("Resuming training from epoch {} at iteration {}".format(start_epoch, steps_per_epoch * start_epoch))
        vgg_netvlad.load_weights(model_name)
        # vgg_netvlad.summary()

    opt = optimizers.Adam(lr=lr)
    vgg_netvlad.compile(opt)

    steps_per_epoch_val = ceil(1491
                               / minibatch_size)

    landmark_generator = my_utils.LandmarkTripletGenerator(train_dir=paths.landmarks_path,
                                                           model=my_model.get_netvlad_extractor(),
                                                           mining_batch_size=mining_batch_size,
                                                           minibatch_size=minibatch_size)

    train_generator = landmark_generator.generator()

    test_generator = my_utils.evaluation_triplet_generator(paths.holidays_small_labeled_path,
                                                           model=my_model.get_netvlad_extractor(),
                                                           netbatch_size=minibatch_size)

    losses = []
    val_losses = []

    not_improving_counter = 0
    not_improving_thresh = 15

    description = train_description

    val_loss_e = []

    for s in range(steps_per_epoch_val):
        x_val, _ = next(test_generator)
        val_loss_s = vgg_netvlad.predict_on_batch(x_val)
        val_loss_e.append(val_loss_s)

    starting_val_loss = np.array(val_loss_e).mean()
    print("Starting validation loss: ", starting_val_loss)
    for e in range(epochs):
        t0 = time.time()

        losses_e = []

        pbar = tqdm(range(steps_per_epoch))

        for s in pbar:
            it = K.get_value(vgg_netvlad.optimizer.iterations)
            if use_warm_up:
                lr = lr_warmup(it, min_lr=max_lr*0.1, max_lr=max_lr, wu_steps=warm_up_steps)
            else:
                lr = max_lr

            K.set_value(vgg_netvlad.optimizer.lr, lr)

            x, y = next(train_generator)
            # print("Starting training at epoch ", e)
            loss_s = vgg_netvlad.train_on_batch(x, y)
            losses_e.append(loss_s)

            description_tqdm = "Loss at epoch {0}/{3} step {1}: {2:.4f}. Lr: {4}".format(e + start_epoch, s, loss_s,
                                                                                         epochs + start_epoch, lr)
            pbar.set_description(description_tqdm)

        print("")
        loss = np.array(losses_e).mean()
        losses.append(loss)

        val_loss_e = []

        for s in range(steps_per_epoch_val):
            x_val, _ = next(test_generator)
            val_loss_s = vgg_netvlad.predict_on_batch(x_val)
            val_loss_e.append(val_loss_s)

        val_loss = np.array(val_loss_e).mean()

        min_val_loss = starting_val_loss

        if e > 0:
            min_val_loss = np.min(val_losses)

        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            model_name = "model_e{0}_{2}_{1:.4f}.h5".format(e + start_epoch, val_loss, description)
            print("Val. loss improved from {0:.4f}. Saving model to: {1}".format(min_val_loss, model_name))
            vgg_netvlad.save_weights(model_name)
            not_improving_counter = 0
        else:
            print("Val loss ({0:.4f}) did not improve from {1:.4f}".format(val_loss, min_val_loss))
            not_improving_counter += 1
            print("Val loss does not improve since {} epochs".format(not_improving_counter))
            if e % 5 == 0:
                model_name = "model_e{0}_{2}_{1:.4f}_checkpoint.h5".format(e + start_epoch, val_loss, description)
                vgg_netvlad.save_weights(model_name)
                print("Saving model to: {} (checkpoint)".format(model_name))
            # if not_improving_counter == not_improving_thresh:
            if False:
                # lr *= 0.5
                # K.set_value(vgg_netvlad.optimizer.lr, lr)
                # print("Learning rate set to: {}".format(lr))
                opt = optimizers.Adam(lr=lr)
                vgg_netvlad.compile(opt)
                not_improving_counter = 0
                print("Optimizer weights restarted.")

        print("Validation loss: {}\n".format(val_loss))
        t1 = time.time()
        print("Time for epoch {}: {}s".format(e, int(t1 - t0)))

    landmark_generator.loader.stop_loading()

    model_name = "model_e{}_{}_.h5".format(epochs + start_epoch, description)
    vgg_netvlad.save_weights(model_name)
    print("Saved model to disk: ", model_name)

    plt.figure(figsize=(8, 8))
    plt.plot(losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.savefig("train_val_loss_{}.pdf".format(description))
    # plt.show()

print("Testing model")
print("Input shape: ", input_shape)

if test and model_name is not None:
    print("Loading ", model_name)
    vgg_netvlad.load_weights(model_name)

vgg_netvlad = my_model.get_netvlad_extractor()

pca_from_landmarks = False
use_pca = False

if pca_from_landmarks and use_pca:
    generator = my_utils.LandmarkTripletGenerator(paths.landmarks_path, model=my_model.get_netvlad_extractor(),
                                                  use_multiprocessing=False)
    custom_generator = generator.generator()

    a = []
    p = []
    n = []

    for i in range(500):
        x, _ = next(custom_generator)
        a_, p_, n_ = vgg_netvlad.predict(x)
        a += [a_]
        p += [p_]
        n += [n_]
        print(i)

    generator.loader.stop_loading()

    a = np.vstack((d for d in a)).astype('float32')
    p = np.vstack((d for d in n)).astype('float32')
    n = np.vstack((d for d in n)).astype('float32')

    descs = np.vstack((a, p, n))
    print(descs.shape)
    del a, p, n

    print("Computing PCA")
    pca = PCA(512)
    pca.fit(descs)
    del descs

    pca_dataset = h5py.File("pca.h5", 'w')
    pca_dataset.create_dataset('components', data=pca.components_)
    pca_dataset.create_dataset('mean', data=pca.mean_)
    pca_dataset.close()

imnames = hth.get_imlist_()
query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]
print('tot images = %d, query images = %d' % (len(imnames), len(query_imids)))

base_resolution = (336, 336, 3)

input_shape_1 = (768, 768, 3)
input_shape_2 = (504, 504, 3)
input_shape_3 = (224, 224, 3)

input_shapes = [input_shape_1, input_shape_2, input_shape_3]

print("Loading images")
img_tensor = hth.create_image_dict(hth.get_imlist(paths.holidays_pic_path), input_shape=base_resolution,
                                   rotate=rotate_holidays)
print("Extracting features")
all_feats = vgg_netvlad.predict(img_tensor, verbose=1)

if use_multi_resolution:
    for shape in input_shapes:
        img_tensor = hth.create_image_dict(hth.get_imlist(paths.holidays_pic_path), input_shape=shape,
                                           rotate=True)
        batch_size = 32
        if shape[0] == 768:
            batch_size = 16

        all_feats += vgg_netvlad.predict(img_tensor, verbose=1, batch_size=batch_size)

if use_power_norm:
    all_feats_sign = np.sign(all_feats)
    all_feats = np.power(np.abs(all_feats), 0.5)
    all_feats = np.multiply(all_feats, all_feats_sign)

all_feats = normalize(all_feats)

query_feats = all_feats[query_imids]

nbrs = NearestNeighbors(n_neighbors=1491, metric='cosine').fit(all_feats)
distances, indices = nbrs.kneighbors(query_feats)

print('mean AP = %.3f' % hth.mAP(query_imids, indices, imnames=imnames))
perfect_result = hth.make_perfect_holidays_result(imnames, query_imids)
print('Perfect mean AP = %.3f' % hth.mAP(query_imids, perfect_result, imnames=imnames))
