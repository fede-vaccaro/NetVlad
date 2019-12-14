# %%
import os
import time
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import open_dataset_utils as my_utils
from netvlad_model import input_shape
from triplet_loss import TripletLossLayer

index, classes = my_utils.generate_index_mirflickr('mirflickr_annotations')

batch_size = 2048
epochs = 40

mirflickr_path = "/mnt/sdb-seagate/datasets/mirflickr/"
files = [mirflickr_path + k for k in list(index.keys())]

# generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=149)
from netvlad_model import NetVLADSiameseModel  # , NetVLADModelRetinaNet

my_model = NetVLADSiameseModel()
vgg, output_shape = my_model.get_feature_extractor(verbose=True)

generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=256)

test_generator, samples_test, _ = my_utils.custom_generator_from_keras("holidays_small_", batch_size=200,
                                                                       net_output=my_model.netvlad_output,
                                                                       train_classes=500)
# k_means_train_generator, _, _ = my_utils.custom_generator_from_keras("/mnt/m2/dataset/", batch_size=200)

vgg_netvlad = my_model.build_netvladmodel()
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
    print("Predicting local features for k-means. Output shape: ", output_shape)
    all_descs = vgg.predict_generator(generator=generator_nolabels, steps=30, verbose=1)
    print("All descs shape: ", all_descs.shape)
    # %%

    import random

    locals = np.vstack((m[np.random.randint(len(m), size=100)] for m in all_descs)).astype('float32')

    print("Sampling local features")
    # %%

    from sklearn.preprocessing import normalize

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

if train:
    # path = "/mnt/sdb-seagate/weights/weights-netvlad-13-03.hdf5"
    # vgg_netvlad.load_weights(path)

    vgg_netvlad.summary()

    # train session
    opt = Adam(lr=0.00001)  # choose optimiser. RMS is good too!

    vgg_netvlad = Model(vgg_netvlad.input, TripletLossLayer(0.1)(vgg_netvlad.output))
    vgg_netvlad.compile(optimizer=opt)

    steps_per_epoch = 50
    steps_per_epoch_val = ceil(samples_test / batch_size)

    filepath = "/mnt/sdb-seagate/weights/weights-netvlad-{epoch:02d}.hdf5"

    landmark_generator = my_utils.LandmarkTripletGenerator(train_dir="/mnt/m2/dataset/",
                                                           model=my_model.get_netvlad_extractor(),
                                                           batch_size=batch_size, net_batch_size=32)

    train_generator = landmark_generator.generator()

    test_generator = my_utils.holidays_triplet_generator("holidays_small_", model=my_model.get_netvlad_extractor(),
                                                         netbatch_size=32)

    losses = []
    val_losses = []

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

# %%


vgg_netvlad.load_weights("model_e204_1212.h5")
vgg_netvlad = my_model.get_netvlad_extractor()
vgg_netvlad.summary()

from keras import layers

## defining second scale
shape2 = (362, 362, 3)
input2 = layers.Input(shape2)
model2 = vgg_netvlad([input2])
model2 = Model(input2, model2)

## defining third scale
shape3 = (256, 256, 3)
input3 = layers.Input(shape3)
model3 = vgg_netvlad([input3])
model3 = Model(input3, model3)

## defining fourth scale
shape4 = (181, 181, 3)
input4 = layers.Input(shape4)
model4 = vgg_netvlad([input4])
model4 = Model(input4, model4)

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


imnames = get_imlist_()

query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]

# check that everything is fine - expected output: "tot images = 1491, query images = 500"
print('tot images = %d, query images = %d' % (len(imnames), len(query_imids)))


# %%

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def create_image_dict(img_list, input_shape):
    # input_shape = (224, 224, 3)
    tensor = {}
    for i, path in enumerate(img_list):
        img = image.load_img(path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img_key = path.strip('holidays/')
        tensor[img_key] = img
        if i % 100 == 0:
            print("Image i loaded: ", i)
    # tensor = np.array(tensor)
    return tensor


# %%

##### FIRST SCALE
print("Computing first scale features")
img_dict = create_image_dict(get_imlist('holidays'), input_shape=input_shape)

img_tensor = [img_dict[key] for key in img_dict]
img_tensor = np.array(img_tensor)

all_feats = vgg_netvlad.predict(img_tensor)
del img_tensor
#####

# ##### SECOND SCALE
# print("Computing second scale features")
# img_dict = create_image_dict(get_imlist('holidays_small'), input_shape=shape2)
#
# # img_tensor = images_to_tensor(imnames)
# img_tensor = [img_dict[key] for key in img_dict]
# img_tensor = np.array(img_tensor)
#
# all_feats_2 = model2.predict(img_tensor)
# del img_tensor
#####

# ##### THIRD SCALE
# print("Computing third scale features")
# img_dict = create_image_dict(get_imlist('holidays_small'), input_shape=shape3)
#
# # img_tensor = images_to_tensor(imnames)
# img_tensor = [img_dict[key] for key in img_dict]
# img_tensor = np.array(img_tensor)
#
# all_feats_3 = model3.predict(img_tensor)
# del img_tensor
# #####
#
# #### FOURTH SCALE
# print("Computing fourth scale features")
# img_dict = create_image_dict(get_imlist('holidays_small'), input_shape=shape4)
#
# # img_tensor = images_to_tensor(imnames)
# img_tensor = [img_dict[key] for key in img_dict]
# img_tensor = np.array(img_tensor)
#
# all_feats_4 = model4.predict(img_tensor)
# del img_tensor
####

# all_feats = normalize(all_feats + sqrt(2) * all_feats_2 + 2 * all_feats_3)

def power_normalization(x, p=0.5):
    x = np.sign(x)*np.abs(x)**p
    x = normalize(x)
    return x
print("")
pca = PCA(512, svd_solver='full')
pca.fit(all_feats)

all_feats = pca.transform(all_feats)
# all_feats_2 = pca.transform(all_feats_2)
# all_feats_3 = pca.transform(all_feats_3)
# all_feats_4 = pca.transform(all_feats_4)

all_feats = power_normalization(all_feats) # + power_normalization(all_feats_2)
all_feats = normalize(all_feats)

# all_feats = pca.transform(all_feats)
# all_feats_sign = np.sign(all_feats)
# all_feats = np.power(np.abs(all_feats), 0.5)
# all_feats = np.multiply(all_feats, all_feats_sign)
# all_feats = normalize(all_feats)

# all_feats = all_feats[:, n_queries:]

# plt.imshow(all_feats, cmap='viridis')
# plt.colorbar()
# plt.grid(False)
# plt.show()

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

# show_result(indices, nqueries=200)
