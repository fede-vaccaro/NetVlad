# %%
import os
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import time
import open_dataset_utils as my_utils
from netvlad_model import input_shape

# index, classes = my_utils.generate_index_holidays('labeled.dat')
# files = my_utils.get_imlist('holidays_small')
index, classes = my_utils.generate_index_mirflickr('mirflickr_annotations')

batch_size = 2048
epochs = 40

mirflickr_path = "/mnt/sdb-seagate/datasets/mirflickr/"
files = [mirflickr_path + k for k in list(index.keys())]

# index, classes = my_utils.generate_index_ukbench('ukbench')
# files = ["ukbench/" + k for k in list(index.keys())]

"""
def comparator(filename):
    key = filename[len("ukbench/ukbench"):]
    key = key[:-len(".jpg")]
    key = int(key)
    return key

files = sorted(files, key=comparator)
"""

# generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=149)
from netvlad_model import NetVLADSiameseModel  # , NetVLADModelRetinaNet

my_model = NetVLADSiameseModel()
vgg, output_shape = my_model.get_feature_extractor(verbose=True)

generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=256)

# train_generator, samples_train, n_classes = my_utils.custom_generator_from_keras("partition_0", batch_size=batch_size, net_output=my_model.netvlad_output, train_classes=None)

test_generator, samples_test, _ = my_utils.custom_generator_from_keras("holidays_small_", batch_size=200,
                                                                       net_output=my_model.netvlad_output,
                                                                       train_classes=500)
# k_means_train_generator, _, _ = my_utils.custom_generator_from_keras("/mnt/m2/dataset/", batch_size=200)

vgg_netvlad = my_model.build_netvladmodel()
# p = vgg_netvlad.predict([np.zeros((1, 224, 224, 3))] * 3)
images = []
for el in generator_nolabels:
    x_batch = el
    images = x_batch
    break
print("features shape: ", images.shape)

train_kmeans = False
train = False

import gc

if train_kmeans:
    # my_model = NetVLADModel()
    # %%

    print("Predicting local features for k-means. Output shape: ", output_shape)
    all_descs = vgg.predict_generator(generator=generator_nolabels, steps=20, verbose=1)
    print("All descs shape: ", all_descs.shape)
    # %%

    import random

    # all_descs_ = np.transpose(all_descs, axes=(0, 3, 1, 2))
    # all_descs = all_descs.reshape((len(all_descs), all_descs.shape[1]*all_descs.shape[2], all_descs.shape[3]))

    locals = np.vstack((m[np.random.randint(len(m), size=100)] for m in all_descs)).astype('float32')

    print("Sampling local features")
    # locals = locals[ np.random.randint(locals.shape[0], ) ]

    # %%

    from sklearn.preprocessing import normalize

    # locals = np.array(locals, dtype='float32')
    locals = normalize(locals, axis=1)
    np.random.shuffle(locals)
    print("Locals: {}".format(locals.shape))
    # %%
    from sklearn.cluster import MiniBatchKMeans

    n_clust = 64
    print("Fitting k-means")
    kmeans = MiniBatchKMeans(n_clusters=n_clust).fit(locals[locals.shape[0] // 4:])

    my_model.set_netvlad_weights(kmeans)

    del all_descs
    gc.collect()

    vgg_netvlad.save_weights("kmeans_weights.h5")

# %%


from sklearn.model_selection import train_test_split

files_train, files_test = train_test_split(files, test_size=0.3, shuffle=False)

# train_generator = my_utils.image_generator(files=files_train, index=index, classes=classes, net_output=my_model.netvlad_output, batch_size=batch_size, augmentation=True)
# test_generator = my_utils.image_generator(files=files_test, index=index, classes=classes, net_output=my_model.netvlad_output, batch_size=batch_size)


if train:
    # path = "/mnt/sdb-seagate/weights/weights-netvlad-13-03.hdf5"
    # vgg_netvlad.load_weights(path)
    from triplet_loss import TripletLossLayer
    # from triplet_loss_ import batch_hard_triplet_loss_k
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

    # from triplet_loss_ import batch_hard_triplet_loss_k

    vgg_netvlad.summary()

    # vgg_netvlad.get_layer('model_1').get_layer('block4_conv2').trainable = True
    # vgg_netvlad.get_layer('model_1').get_layer('block4_conv3').trainable = True

    # train session
    opt = Adam(lr=0.00001)  # choose optimiser. RMS is good too!

    # loss_layer = TripletLossLayer(alpha=1., name='triplet_loss_layer')(vgg_netvlad.output)
    # vgg_qpn = Model(inputs=vgg_qpn.input, outputs=loss_layer)
    vgg_netvlad = Model(vgg_netvlad.input, TripletLossLayer(0.1)(vgg_netvlad.output))
    vgg_netvlad.compile(optimizer=opt)

    # steps_per_epoch = ceil(samples_train / batch_size)
    steps_per_epoch = 50
    steps_per_epoch_val = ceil(samples_test / batch_size)

    filepath = "/mnt/sdb-seagate/weights/weights-netvlad-{epoch:02d}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min',
                                 save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=10, min_lr=1e-9, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='min',
                                   baseline=None, restore_best_weights=False)

    # from keras import backend as K
    # K.set_value(vgg_netvlad.optimizer.lr, 0.1) # warm up
    # vgg_netvlad.fit_generator(generator=generator,
    #                              steps_per_epoch=10,
    #                              epochs=3, use_multiprocessing=False)
    # print("warmed up")

    train_generator = my_utils.landmark_triplet_generator("/mnt/m2/dataset/", model=my_model.get_netvlad_extractor(),
                                                          batch_size=batch_size, netbatch_size=32)

    test_generator = my_utils.holidays_triplet_generator("holidays_small_", model=my_model.get_netvlad_extractor(),
                                                         netbatch_size=32)

    # K.set_value(vgg_netvlad.optimizer.lr, 0.0001) # warm up
    """
    H = vgg_netvlad.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_steps=ceil(1491/batch_size),
                                  validation_data=test_generator,
                                  callbacks=[reduce_lr, checkpoint], use_multiprocessing=False, max_queue_size=1)
    """
    losses = []
    val_losses = []

    for e in range(epochs):
        t0 = time.time()
        for s in range(steps_per_epoch):
            x, y = next(train_generator)
            loss = vgg_netvlad.train_on_batch(x, y)
            losses.append(loss)
            print("Loss at epoch {} step {}: {}\n".format(e, s, loss))

        val_loss_e = []

        for s in range(ceil(1491 / 32)):
            x_val, _ = next(test_generator)
            val_loss_s = vgg_netvlad.predict_on_batch(x_val)
            val_loss_e.append(val_loss_s)

        val_loss = np.array(val_loss_e).mean()

        min_val_loss = np.inf

        if e > 0:
            min_val_loss = np.min(val_losses)

        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            model_name = "model_e{}.h5".format(e)
            print("Val. loss improved from {}. Saving model to: {}".format(min_val_loss, model_name))
            vgg_netvlad.save_weights(model_name)
        else:
            print("Val loss ({}) did not improved from {}".format(val_loss, min_val_loss))
             # val_losses.append(val_loss)

        print("Validation loss: {}\n".format(val_loss))
        t1 = time.time()
        print("Time for epoch {}: {}s".format(e, int(t1 - t0)))

    vgg_netvlad.save_weights("model.h5")
    print("Saved model to disk")

    plt.figure(figsize=(8, 8))
    # plt.plot(H.history['loss'], label='training loss')
    # plt.plot(H.history['val_loss'], label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.show()

# %%
vgg_netvlad.load_weights("model_e35_891.h5")
vgg_netvlad = my_model.get_netvlad_extractor()
vgg_netvlad.summary()
result = vgg_netvlad.predict(images[:1])
# %%

print("Testing model")

############### test model
# this function create a perfect ranking :)
from sklearn.neighbors import NearestNeighbors


# input_shape = (224, 224, 3)


def get_imlist_(path="holidays_small"):
    imnames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]
    imnames = [path.strip('holidays_small/') for path in imnames]
    imnames = [path.strip('.jpg') for path in imnames]
    return imnames


def images_to_tensor(imnames):
    images_array = []

    # open all images
    for name in imnames:
        img_path = 'holidays_small/' + name + '.jpg'
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        images_array.append(img)
    images_array = np.array(images_array)
    print(images_array.shape)
    # images_array = preprocess_input(images_array)
    return images_array


imnames = get_imlist_()

query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]

# check that everything is fine - expected output: "tot images = 1491, query images = 500"
print('tot images = %d, query images = %d' % (len(imnames), len(query_imids)))


# %%

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def create_image_dict(img_list):
    # input_shape = (224, 224, 3)
    tensor = {}
    for path in img_list:
        img = image.load_img(path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img_key = path.strip('holidays_small/')
        tensor[img_key] = img
    # tensor = np.array(tensor)
    return tensor


# %%

img_dict = create_image_dict(get_imlist('holidays_small'))
img_dict.keys()

# img_tensor = images_to_tensor(imnames)
img_tensor = [img_dict[key] for key in img_dict]
img_tensor = np.array(img_tensor)

# %%

# %%
# vgg_netvlad.summary()
all_feats = vgg_netvlad.predict(img_tensor)

print("")
# all_feats = PCA(512, svd_solver='full').fit_transform(all_feats)
all_feats_sign = np.sign(all_feats)
all_feats = np.power(np.abs(all_feats), 0.5)
all_feats = np.multiply(all_feats, all_feats_sign)
all_feats = normalize(all_feats)

# all_feats = all_feats[:, n_queries:]

plt.imshow(all_feats, cmap='viridis')
plt.colorbar()
plt.grid(False)
plt.show()

# %%

query_feats = all_feats[query_imids]

# SOLUTION
nbrs = NearestNeighbors(n_neighbors=1491, metric='cosine').fit(all_feats)
distances, indices = nbrs.kneighbors(query_feats)


# %%

def make_perfect_holidays_result(imnames, q_ids):
    perfect_idx = []
    for qimno in q_ids:
        qname = imnames[qimno]
        positive_results = set([i for i, name in enumerate(imnames) if name != qname and name[:4] == qname[:4]])
        ok = [qimno] + [i for i in positive_results]
        others = [i for i in range(1491) if i not in positive_results and i != qimno]
        perfect_idx.append(ok + others)
    return np.array(perfect_idx)


def mAP(q_ids, idx):
    aps = []
    for qimno, qres in zip(q_ids, idx):
        qname = imnames[qimno]
        # collect the positive results in the dataset
        # the positives have the same prefix as the query image
        positive_results = set([i for i, name in enumerate(imnames)
                                if name != qname and name[:4] == qname[:4]])
        #
        # ranks of positives. We skip the result #0, assumed to be the query image
        ranks = [i for i, res in enumerate(qres[1:]) if res in positive_results]
        #
        # accumulate trapezoids with this basis
        recall_step = 1.0 / len(positive_results)
        ap = 0
        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far
            # y-size on left side of trapezoid:
            precision_0 = ntp / float(rank) if rank > 0 else 1.0
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        # print('query %s, AP = %.3f' % (qname, ap))
        aps.append(ap)
    return np.mean(aps)


print('mean AP = %.3f' % mAP(query_imids, indices))
perfect_result = make_perfect_holidays_result(imnames, query_imids)
print('Perfect mean AP = %.3f' % mAP(query_imids, perfect_result))

import PIL

import math


def montage(imfiles, thumb_size=(100, 100), ok=None, shape=None):
    # this function will create an image with thumbnailed version of imfiles.
    # optionally the user can provide an ok list such that len(ok)==len(imfiles) to differentiate correct from wrong results
    # optionally the user can provide a shape function which shapes the montage otherwise a square image is created.
    images = [PIL.Image.open(imname).resize(thumb_size, PIL.Image.BILINEAR) for imname in imfiles]
    # create a big image to contain all images
    if shape is None:
        n = int(math.sqrt(len(imfiles)))
        m = n
    else:
        n = shape[0]
        m = shape[1]
    new_im = PIL.Image.new('RGB', (m * thumb_size[0], n * thumb_size[0]))
    k = 0
    for i in range(0, n * thumb_size[0], thumb_size[0]):
        for j in range(0, m * thumb_size[0], thumb_size[0]):
            region = (j, i)
            if ok is not None:
                if ok[k]:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                if k > 0:
                    imar = np.array(images[k], dtype=np.uint8)
                    imar[0:5, :, :] = color
                    imar[:, 0:5, :] = color
                    imar[-5:, :, :] = color
                    imar[:, -5:, :] = color
                    images[k] = PIL.Image.fromarray(imar)
            new_im.paste(images[k], box=region)
            k += 1
    return new_im


# %%

# here we show the first 25 queries and their 15 closest neighbours retrieved
# gree border means ok, red wrong :)
def show_result(display_idx, nqueries=10, nresults=10, ts=(100, 100)):
    if nqueries is not None:
        nrow = nqueries  # number of query images to show

    if nresults is not None:
        nres = 10  # number of results per query

    for qno in range(nrow):
        imfiles = []
        oks = [True]
        # show query image with white outline
        qimno = query_imids[qno]
        imfiles.append('holidays_small/' + imnames[qimno] + '.jpg')
        for qres in display_idx[qno, :nres]:
            # use image name to determine if it is a TP or FP result
            oks.append(imnames[qres][:4] == imnames[qimno][:4])
            imfiles.append('holidays_small/' + imnames[qres] + '.jpg')
        # print(qno, (imfiles))
        plt.imshow(montage(imfiles, thumb_size=ts, ok=oks, shape=(1, nres)))
        plt.show()


# %%

show_result(indices, nqueries=200)
