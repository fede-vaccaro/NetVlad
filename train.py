# %%
import gc
import time
from math import ceil
import h5py

import matplotlib.pyplot as plt
import numpy as np
from keras import Model, optimizers
from keras import backend as K
from keras.models import load_model
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import holidays_testing_helpers as hth
import open_dataset_utils as my_utils
from triplet_loss import TripletLossLayer
import triplet_loss as tl
import loupe_keras as lk
from netvlad_model import NetVLADSiameseModel, NetVladResnet
import paths
import scipy
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
ap.add_argument("-t", "--test", action='store_true',
                help="If must be bypassed the training for testing")
args = vars(ap.parse_args())

mining_batch_size = 2048
minibatch_size = 24
epochs = 60

index, classes = my_utils.generate_index_mirflickr(paths.mirflickr_annotations)
files = [paths.mirflickr_path + k for k in list(index.keys())]

my_model = NetVLADSiameseModel()
vgg, output_shape = my_model.get_feature_extractor(verbose=True)

generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=160)
vgg_netvlad = my_model.build_netvladmodel()

print("Netvlad output shape: ", vgg_netvlad.output_shape)
print("Feature extractor output shape: ", vgg.output_shape)

test = args['test']
train_kmeans = False and not test
train = True and not test

if train_kmeans:
    print("Predicting local features for k-means. Output shape: ", output_shape)
    all_descs = vgg.predict_generator(generator=generator_nolabels, steps=45, verbose=1)
    print("All descs shape: ", all_descs.shape)

    locals = np.vstack((m[np.random.randint(len(m), size=150)] for m in all_descs)).astype('float32')

    print("Sampling local features")

    locals = normalize(locals, axis=1)
    np.random.shuffle(locals)
    print("Locals extracted: {}".format(locals.shape))

    n_clust = 64
    print("Fitting k-means")
    kmeans = MiniBatchKMeans(n_clusters=n_clust).fit(locals[locals.shape[0] // 3:])

    my_model.set_netvlad_weights(kmeans)

    del all_descs
    gc.collect()

if train:
    vgg_netvlad.summary()

    start_epoch = args['start_epoch']
    vgg_netvlad = Model(vgg_netvlad.input, TripletLossLayer(0.1)(vgg_netvlad.output))
    lr = 0.001
    # opt = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)  # choose optimiser. RMS is good too!
    opt = optimizers.Adam(lr=lr)  # choose optimiser. RMS is good too!
    vgg_netvlad.compile(opt)

    if args['model'] is not None:
        model_name = args['model']
        print("Resuming training from epoch {}".format(start_epoch))
        vgg_netvlad.load_weights(model_name)
        # vgg_netvlad = load_model(model_name, custom_objects={"L2NormLayer": tl.L2NormLayer, "NetVLAD": lk.NetVLAD,
        #                                                     "TripletLossLayer": TripletLossLayer})
        K.set_value(vgg_netvlad.optimizer.lr, 1e-6)
        print("LR set to: ", K.get_value(vgg_netvlad.optimizer.lr))

    steps_per_epoch = 50
    steps_per_epoch_val = ceil(1491 / minibatch_size)

    landmark_generator = my_utils.LandmarkTripletGenerator(train_dir=paths.landmarks_path,
                                                           model=my_model.get_netvlad_extractor(),
                                                           mining_batch_size=mining_batch_size,
                                                           minibatch_size=minibatch_size)

    train_generator = landmark_generator.generator()

    test_generator = my_utils.holidays_triplet_generator(paths.holidays_small_labeled_path,
                                                         model=my_model.get_netvlad_extractor(),
                                                         netbatch_size=minibatch_size)

    losses = []
    val_losses = []

    not_improving_counter = 0
    not_improving_thresh = 15
    for e in range(epochs):
        t0 = time.time()

        losses_e = []

        for s in range(steps_per_epoch):
            x, y = next(train_generator)
            loss_s = vgg_netvlad.train_on_batch(x, y)
            losses_e.append(loss_s)
            print("Loss at epoch {0} step {1}: {2:.4f}".format(e + start_epoch, s, loss_s))

        print("")
        loss = np.array(losses_e).mean()
        losses.append(loss)

        val_loss_e = []

        for s in range(steps_per_epoch_val):
            x_val, _ = next(test_generator)
            val_loss_s = vgg_netvlad.predict_on_batch(x_val)
            val_loss_e.append(val_loss_s)

        val_loss = np.array(val_loss_e).mean()

        min_val_loss = 0.0681

        if e > 0:
            min_val_loss = np.min(val_losses)

        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            model_name = "model_e{0}_adam_l{1:.4f}_.h5".format(e + start_epoch, val_loss)
            print("Val. loss improved from {0:.4f}. Saving model to: {1}".format(min_val_loss, model_name))
            vgg_netvlad.save(model_name)
            not_improving_counter = 0
        else:
            print("Val loss ({0:.4f}) did not improved from {1:.4f}".format(val_loss, min_val_loss))
            not_improving_counter += 1
            print("Val loss does not improve since {} epochs".format(not_improving_counter))
            # if not_improving_counter == not_improving_thresh:
            if e % 5 == 0:
                model_name = "model_e{0}_adam_l{1:.4f}_checkpoint.h5".format(e + start_epoch, val_loss)
                print("Saving model to: {} (checkpoint)".format(model_name))
            if False:
                lr *= 0.5
                K.set_value(vgg_netvlad.optimizer.lr, lr)
                print("Learning rate set to: {}".format(lr))
                not_improving_counter = 0

        print("Validation loss: {}\n".format(val_loss))
        t1 = time.time()
        print("Time for epoch {}: {}s".format(e, int(t1 - t0)))

    landmark_generator.loader.stop_loading()

    model_name = "model_e{}_1812_adam.h5".format(epochs + start_epoch)
    vgg_netvlad.save(model_name)
    print("Saved model to disk")

    plt.figure(figsize=(8, 8))
    plt.plot(losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.savefig("train_val_loss.pdf")
    # plt.show()

print("Testing model")

# path = "model_e122_1812_sgd_l0.0712_.h5"
# vgg_netvlad.load_weights(path)
vgg_netvlad = my_model.get_netvlad_extractor()
pca_from_landmarks = False
if pca_from_landmarks:
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
# print(imnames)
# check that everything is fine - expected output: "tot images = 1491, query images = 500"
print('tot images = %d, query images = %d' % (len(imnames), len(query_imids)))

print("Loading images")
img_dict = hth.create_image_dict(hth.get_imlist(paths.holidays_pic_path), rotate=True)


img_tensor = [img_dict[key] for key in img_dict]
img_tensor = np.array(img_tensor)

print("Extracting features")
all_feats = vgg_netvlad.predict(img_tensor)

print("Training PCA")
if pca_from_landmarks:
    all_feats = pca.transform(all_feats)
else:
    pca = PCA(512, svd_solver='full')
    # dataset = h5py.File("pca.h5", 'r')
    # components = dataset['components'][:]
    # mean = dataset['mean'][:]
    # pca.components_ = components
    # pca.mean_ = mean
    pca.fit(all_feats)
    all_feats = pca.transform(all_feats)

    # all_feats = PCA(512, svd_solver='full').fit_transform(all_feats)

all_feats_sign = np.sign(all_feats)
all_feats = np.power(np.abs(all_feats), 0.5)
all_feats = np.multiply(all_feats, all_feats_sign)
all_feats = normalize(all_feats)

# all_feats = all_feats[:, n_queries:]

# plt.imshow(all_feats, cmap='viridis')
# plt.colorbar()
# plt.grid(False)
# plt.show()

query_feats = all_feats[query_imids]

nbrs = NearestNeighbors(n_neighbors=1491, metric='cosine').fit(all_feats)
distances, indices = nbrs.kneighbors(query_feats)

print('mean AP = %.3f' % hth.mAP(query_imids, indices, imnames=imnames))
perfect_result = hth.make_perfect_holidays_result(imnames, query_imids)
print('Perfect mean AP = %.3f' % hth.mAP(query_imids, perfect_result, imnames=imnames))

# hth.show_result(indices, nqueries=200)
