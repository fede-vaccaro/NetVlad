# %%
import argparse
import gc
import os
import time
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import yaml
from keras import optimizers, Model
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
# from keras_radam import RAdam
# from keras_radam.training import RAdamOptimizer
from tqdm import tqdm
import utils
import clr_callback as clr
import holidays_testing_helpers as hth
import netvlad_model
import open_dataset_utils as my_utils
import paths
from triplet_loss import TripletLossLayer
import h5py

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

conf_file = open(config_file, 'r')
conf = dict(yaml.safe_load(conf_file))
conf_file.close()

# network
network_conf = conf['network']
net_name = network_conf['name']

n_cluster = network_conf['n_clusters']
middle_pca = network_conf['middle_pca']

# mining
try:
    threshold = conf['threshold']
except:
    threshold = 20

try:
    semi_hard_prob = conf['semi-hard-prob']
except:
    semi_hard_prob = 0.5

# training
train_description = conf['description']
mining_batch_size = conf['mining_batch_size']
minibatch_size = conf['minibatch_size']
steps_per_epoch = conf['steps_per_epoch']
epochs = conf['n_epochs']

# learning rate
lr_conf = conf['lr']
use_warm_up = lr_conf['warm-up']
warm_up_steps = lr_conf['warm-up-steps']
max_lr = float(lr_conf['max_value'])

# testing
rotate_holidays = conf['rotate_holidays']
use_power_norm = conf['use_power_norm']
use_multi_resolution = conf['use_multi_resolution']

side_res = conf['input-shape']

netvlad_model.NetVladBase.input_shape = (side_res, side_res, 3)
if use_multi_resolution:
    netvlad_model.NetVladBase.input_shape = (None, None, 3)

# if test:
#     gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#     for device in gpu_devices:
#         tf.config.experimental.set_memory_growth(device, True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

my_model = None
if net_name == "vgg":
    my_model = netvlad_model.NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    # my_model = netvlad_model.NetVladResnet(**network_conf)
    my_model = netvlad_model.GeMResnet(**network_conf)
else:
    print("Network name not valid.")


vgg, output_shape = my_model.get_feature_extractor(verbose=True)

vgg_netvlad = my_model.build_netvladmodel()
vgg_netvlad.summary()

print("Netvlad output shape: ", vgg_netvlad.output_shape)
print("Feature extractor output shape: ", vgg.output_shape)

train_pca = False
# train kmeans disabled for GeM
train_kmeans = (not test or test_kmeans) and model_name is None and not train_pca
train = not test

if test and model_name is not None:
    print("Loading ", model_name)
    vgg_netvlad.load_weights(model_name)

backbone = my_model.get_netvlad_extractor()


my_model_netvlad = netvlad_model.NetVladResnet(**network_conf)
my_model_netvlad.build_base_model(backbone=Model(backbone.input, backbone.get_layer('add_16').output))
my_model_netvlad.build_netvladmodel()

my_model = my_model_netvlad

if train_kmeans:
    kmeans_generator = image.ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        paths.landmarks_path,
        target_size=(netvlad_model.NetVladBase.input_shape[0], netvlad_model.NetVladBase.input_shape[1]),
        batch_size=128,
        class_mode=None,
        interpolation='bilinear', seed=4242)

    print("Predicting local features for k-means. Output shape: ", output_shape)
    all_descs = vgg.predict_generator(generator=kmeans_generator, steps=30, verbose=1)
    print("All descs shape: ", all_descs.shape)

    locals = np.vstack((m[np.random.randint(len(m), size=150)] for m in all_descs)).astype('float32')

    print("Sampling local features")

    np.random.shuffle(locals)

    if middle_pca['pretrain'] and middle_pca['active']:
        print("Training PCA")
        pca = PCA(middle_pca['dim'])
        locals = pca.fit_transform(locals)
        my_model.set_mid_pca_weights(pca)
    print("Locals extracted: {}".format(locals.shape))

    n_clust = my_model.n_cluster

    locals = normalize(locals, axis=1)

    print("Fitting k-means")
    kmeans = MiniBatchKMeans(n_clusters=n_clust).fit(locals)

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

    kmeans_generator = my_utils.LandmarkTripletGenerator(train_dir=paths.landmarks_path,
                                                         model=my_model.get_netvlad_extractor(),
                                                         mining_batch_size=mining_batch_size,
                                                         minibatch_size=minibatch_size, semi_hard_prob=semi_hard_prob,
                                                         threshold=threshold)

    train_generator = kmeans_generator.generator()

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
                lr = utils.lr_warmup(it, min_lr=max_lr * 0.1, max_lr=max_lr, wu_steps=warm_up_steps)
            else:
                lr = max_lr

            K.set_value(vgg_netvlad.optimizer.lr, lr)

            x, y = next(train_generator)
            # print("Starting training at epoch ", e)
            loss_s = vgg_netvlad.train_on_batch(x, None)
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

    kmeans_generator.loader.stop_loading()

    model_name = "model_e{}_{}_.h5".format(epochs + start_epoch, description)
    vgg_netvlad.save_weights(model_name)
    print("Saved model to disk: ", model_name)

    plt.figure(figsize=(8, 8))
    plt.plot(losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.savefig("train_val_loss_{}.pdf".format(description))

print("Testing model")
print("Input shape: ", netvlad_model.NetVladBase.input_shape)


vgg_netvlad = my_model.get_netvlad_extractor()

imnames = hth.get_imlist_()
query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]
print('tot images = %d, query images = %d' % (len(imnames), len(query_imids)))

base_resolution = (side_res, side_res, 3)

input_shape_1 = (768, 768, 3)
input_shape_2 = (504, 504, 3)
input_shape_3 = (224, 224, 3)

input_shapes = [input_shape_1, input_shape_2, input_shape_3]

print("Loading images")
img_tensor = hth.create_image_dict(hth.get_imlist(paths.holidays_pic_path), input_shape=base_resolution,
                                   rotate=rotate_holidays)
print("Extracting features")
all_feats = vgg_netvlad.predict(img_tensor, verbose=1, batch_size=3)

if use_multi_resolution:
    for shape in input_shapes:
        img_tensor = hth.create_image_dict(hth.get_imlist(paths.holidays_pic_path), input_shape=shape,
                                           rotate=True)
        batch_size = 32
        if shape[0] >= 768:
            batch_size = 12

        all_feats += vgg_netvlad.predict(img_tensor, verbose=1, batch_size=batch_size)

all_feats = normalize(all_feats)

use_pca = False
if use_pca:
    n_components = 2048

    pca_dataset = h5py.File("pca_{}.h5".format(n_components), 'r')
    mean = pca_dataset['mean'][:]
    components = pca_dataset['components'][:]
    explained_variance = pca_dataset['explained_variance'][:]
    pca_dataset.close()

    all_feats = utils.transform(all_feats, mean, components, explained_variance, whiten=True, pow_whiten=0.5)

if use_power_norm:
    all_feats_sign = np.sign(all_feats)
    all_feats = np.power(np.abs(all_feats), 0.5)
    all_feats = np.multiply(all_feats, all_feats_sign)


query_feats = all_feats[query_imids]

nbrs = NearestNeighbors(n_neighbors=1491, metric='cosine').fit(all_feats)
distances, indices = nbrs.kneighbors(query_feats)

print('mean AP = %.3f' % hth.mAP(query_imids, indices, imnames=imnames))
perfect_result = hth.make_perfect_holidays_result(imnames, query_imids)
print('Perfect mean AP = %.3f' % hth.mAP(query_imids, perfect_result, imnames=imnames))
