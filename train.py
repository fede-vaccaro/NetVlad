# %%
import gc
import time
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from keras import Model, layers
from keras import backend as K
from keras.optimizers import Adam
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import holidays_testing_helpers as hth
import open_dataset_utils as my_utils
from netvlad_model import NetVLADSiameseModel
from triplet_loss import TripletLossLayer

import h5py

mining_batch_size = 2048
minibatch_size = 24
epochs = 40

index, classes = my_utils.generate_index_mirflickr('mirflickr_annotations')
mirflickr_path = "/mnt/sdb-seagate/datasets/mirflickr/"
files = [mirflickr_path + k for k in list(index.keys())]

my_model = NetVLADSiameseModel()
vgg, output_shape = my_model.get_feature_extractor(verbose=True)

generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=256)
vgg_netvlad = my_model.build_netvladmodel()

print("Netvlad output shape: ", vgg_netvlad.output_shape)
print("Feature extractor output shape: ", vgg.output_shape)

train_kmeans = False
train = True

if train_kmeans:
    print("Predicting local features for k-means. Output shape: ", output_shape)
    all_descs = vgg.predict_generator(generator=generator_nolabels, steps=30, verbose=1)
    print("All descs shape: ", all_descs.shape)

    locals = np.vstack((m[np.random.randint(len(m), size=100)] for m in all_descs)).astype('float32')

    print("Sampling local features")

    locals = normalize(locals, axis=1)
    np.random.shuffle(locals)
    print("Locals extracted: {}".format(locals.shape))

    n_clust = 64
    print("Fitting k-means")
    kmeans = MiniBatchKMeans(n_clusters=n_clust).fit(locals[locals.shape[0] // 4:])

    my_model.set_netvlad_weights(kmeans)

    del all_descs
    gc.collect()

vgg_netvlad = my_model.build_siamese_network(vgg_netvlad)

vgg_netvlad.load_weights("model_e219_1212_897.h5")
netvlad_base = my_model.get_netvlad_extractor()

for layer in netvlad_base.layers:
    layer.trainable = False

out = netvlad_base.get_layer('net_vlad_1').output
out = layers.Dropout(0.2)(out)

pca_layer = layers.Dense(512, activation=None)
out = pca_layer(out)
out = layers.Lambda(lambda x_: K.l2_normalize(x_, 0))(out)
netvlad_base = Model(my_model.base_model.input, out)

pca_files = h5py.File('pca.h5', 'r')
components_ = pca_files['components'][:]
mean_ = pca_files['mean'][:]
mean_ = -np.dot(mean_, components_.T)
print(mean_.shape)
pca_layer.set_weights([components_.T, mean_])
pca_files.close()


netvlad_base.summary()

vgg_netvlad = my_model.build_siamese_network(netvlad_base)
vgg_netvlad = Model(vgg_netvlad.input, TripletLossLayer(0.1)(vgg_netvlad.output))
vgg_netvlad.summary()

if train:
    # train session

    lr = 0.00001
    opt = Adam(lr=lr)  # choose optimiser. RMS is good too!

    vgg_netvlad.compile(optimizer=opt)

    steps_per_epoch = 50
    steps_per_epoch_val = ceil(1491 / minibatch_size)

    filepath = "/mnt/sdb-seagate/weights/weights-netvlad-{epoch:02d}.hdf5"

    landmark_generator = my_utils.LandmarkTripletGenerator(train_dir="/mnt/m2/dataset/",
                                                           model=my_model.get_netvlad_extractor(),
                                                           mining_batch_size=mining_batch_size,
                                                           minibatch_size=minibatch_size)

    train_generator = landmark_generator.generator()

    test_generator = my_utils.holidays_triplet_generator("holidays_small_", model=my_model.get_netvlad_extractor(),
                                                         netbatch_size=minibatch_size)


    pretraining = False
    if pretraining:
        pretraining_epochs = 3
        for e in range(pretraining_epochs):
            if e == 0:
                lr = 0.0005
                K.set_value(vgg_netvlad.optimizer.lr, lr)

            for s in range(steps_per_epoch):
                x, y = next(train_generator)
                loss_s = vgg_netvlad.train_on_batch(x, y)
                print("Loss at pre-training epoch {} step {}: {}\n".format(e, s, loss_s))

    losses = []
    val_losses = []

    not_improving_counter = 0
    patience = 20

    lr = 0.00001
    K.set_value(vgg_netvlad.optimizer.lr, lr)

    for e in range(epochs):
        t0 = time.time()

        losses_e = []

        for s in range(steps_per_epoch):
            x, y = next(train_generator)
            loss_s = vgg_netvlad.train_on_batch(x, y)
            losses_e.append(loss_s)
            print("Loss at epoch {} step {}: {}\n".format(e, s, loss_s))

        loss = np.array(losses_e).mean()
        losses.append(loss)

        val_loss_e = []

        for s in range(steps_per_epoch_val):
            x_val, _ = next(test_generator)
            val_loss_s = vgg_netvlad.predict_on_batch(x_val)
            val_loss_e.append(val_loss_s)

        val_loss = np.array(val_loss_e).mean()

        min_val_loss = np.inf

        if e > 0:
            min_val_loss = np.min(val_losses)

        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            model_name = "model_e{}_trained_pca.h5".format(e)
            print("Val. loss improved from {}. Saving model to: {}".format(min_val_loss, model_name))
            vgg_netvlad.save(model_name)
            not_improving_counter = 0
        else:
            print("Val loss ({}) did not improved from {}".format(val_loss, min_val_loss))
            not_improving_counter += 1
            print("Val loss does not improve since {} epochs".format(not_improving_counter))
            if not_improving_counter == patience:
                lr *= 0.5
                K.set_value(vgg_netvlad.optimizer.lr, lr)
                print("Learning rate set to: {}".format(lr))
                not_improving_counter = 0

        print("Validation loss: {}\n".format(val_loss))
        t1 = time.time()
        print("Time for epoch {}: {}s".format(e, int(t1 - t0)))

    landmark_generator.loader.stop_loading()

    vgg_netvlad.save_weights("model.h5")
    print("Saved model to disk")

    plt.figure(figsize=(8, 8))
    plt.plot(losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.savefig("train_val_loss.pdf")
    # plt.show()

print("Testing model")

# vgg_netvlad.load_weights("model.h5")
vgg_netvlad = my_model.get_netvlad_extractor()
vgg_netvlad.summary()

imnames = hth.get_imlist_()
query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]

# check that everything is fine - expected output: "tot images = 1491, query images = 500"
print('tot images = %d, query images = %d' % (len(imnames), len(query_imids)))

img_dict = hth.create_image_dict(hth.get_imlist('holidays_small'))
img_dict.keys()

img_tensor = [img_dict[key] for key in img_dict]
img_tensor = np.array(img_tensor)

all_feats = vgg_netvlad.predict(img_tensor)


print("Computing PCA")
# all_feats = PCA(512, svd_solver='full').fit_transform(all_feats)
# all_feats_sign = np.sign(all_feats)
# all_feats = np.power(np.abs(all_feats), 0.5)
# all_feats = np.multiply(all_feats, all_feats_sign)
#all_feats = normalize(all_feats)

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
