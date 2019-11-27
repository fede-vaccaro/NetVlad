#%%
import tensorflow as tf
import matplotlib.pyplot as plt


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], False)

import os

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from math import ceil
from keras.optimizers import Adam
import random

import open_dataset_utils as my_utils
index, classes = my_utils.generate_index_holidays('labeled.dat')
files = my_utils.get_imlist('holidays_small')
query_holidays = [x for x in files if int(x.strip('holidays_small/')[:-4]) % 100 is 0]
random.shuffle(query_holidays)
query_holidays = query_holidays[:50]
# index, classes = my_utils.generate_index_mirflickr('mirflickr_annotations')
# files = ["mirflickr/" + k for k in list(index.keys())]

from netvlad_model import NetVLADModel, NetVLADModelRetinaNet

my_model = NetVLADModel()
vgg, output_shape = my_model.get_feature_extractor(verbose=True)
vgg_netvlad = my_model.build_netvladmodel(n_classes=50)

generator = my_utils.image_generator(files=query_holidays, index=index, classes=classes, net_output=my_model.netvlad_output, batch_size=500)
generator_nolabels = my_utils.image_generator(files=files, index=index, classes=classes, batch_size=128)

images = []

queries_train = None
labels_train = None

for el in generator:
    [x_batch, label_batch], y_batch = el
    queries_train = x_batch
    labels_train = label_batch
    break

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=70,
                                   width_shift_range=0.5,
                                   height_shift_range=0.5,
                                   shear_range=0.6,
                                   zoom_range=0.6,
                                   horizontal_flip=False,
                                   fill_mode='nearest')

train_kmeans = True
if train_kmeans:

    # my_model = NetVLADModel()
    # %%


    print("Predicting local features for k-means. Output shape: ", output_shape)
    all_descs = vgg.predict_generator(generator=generator_nolabels, steps=15)
    print("All descs shape: ", all_descs.shape)
    # %%

    import random

    #all_descs_ = np.transpose(all_descs, axes=(0, 3, 1, 2))
    all_descs = all_descs.reshape((len(all_descs), all_descs.shape[1]*all_descs.shape[2], all_descs.shape[3]))

    locals = []

    print("Sampling local features")
    for desc_matrix in all_descs:
        samples = random.sample(desc_matrix.tolist(), 25)
        locals += samples

    #%%

    from sklearn.preprocessing import normalize

    locals = np.array(locals, dtype='float32')
    locals = normalize(locals, axis=1)
    #%%
    from sklearn.cluster import MiniBatchKMeans

    n_clust = 64
    print("Fitting k-means")
    kmeans = MiniBatchKMeans(n_clusters=n_clust).fit(locals)

    my_model.set_netvlad_weights(kmeans)

#%%

batch_size = 50
epochs = 70

from sklearn.model_selection import train_test_split

files_train, files_test = train_test_split(files, test_size=0.15, random_state=42)

generator = my_utils.image_generator(files=files_train, index=index, classes=classes, net_output=my_model.netvlad_output, batch_size=batch_size)
test_generator = my_utils.image_generator(files=files_test, index=index, classes=classes, net_output=my_model.netvlad_output, batch_size=batch_size)

print("Loading images")
x_batch = []
label_batch = []
for el in generator:
    [x_batch, label_batch], y_batch = el
    break

#result = vgg_netvlad.predict([x_batch[:1], label_batch[:1]])

train = True
if train:
    from triplet_loss import TripletLossLayer, triplet_loss_adapted_from_tf_multidimlabels
    from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

    def schudule(epoch, lr):
        if epoch + 1 % 3 is 0:
            print("Halving lr to: ")
            lr *= 0.5

        return lr

    # from triplet_loss_ import batch_hard_triplet_loss_k

    # train session
    opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!

    # loss_layer = TripletLossLayer(alpha=1., name='triplet_loss_layer')(vgg_netvlad.output)
    # vgg_qpn = Model(inputs=vgg_qpn.input, outputs=loss_layer)
    vgg_netvlad.compile(optimizer=opt, loss=triplet_loss_adapted_from_tf_multidimlabels)

    steps_per_epoch = ceil(len(files_train) / batch_size)
    steps_per_epoch_val = ceil(len(files_test) / batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=1e-9, verbose=1)

    schedule_lr = LearningRateScheduler(schudule, verbose=1)

    """H = vgg_netvlad.fit_generator(generator=generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_steps=steps_per_epoch_val,
                                  validation_data=test_generator,
                                  callbacks=[reduce_lr, schedule_lr])
    """
    from open_dataset_utils import image_generator_ones
    generator_ones = image_generator_ones(files=files, net_output=my_model.netvlad_output, batch_size=50)
    partitions = []

    n_parts = 10
    i = 0
    for el in generator_ones:
        partitions += [el]
        i += 1
        if i == n_parts:
            break


    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        partition, _ = partitions[e % len(partitions)]
        queries_train = partition[0].tolist()
        labels_train = partition[1].tolist()

        queries_train *= 4
        labels_train *= 4

        queries_train = np.array(queries_train)
        labels_train = np.array(labels_train)

        for x_batch, y_batch in train_datagen.flow(queries_train, labels_train, batch_size=batch_size, shuffle=True):
            vgg_netvlad.fit([x_batch, y_batch], np.zeros((len(x_batch), my_model.netvlad_output + 50)), batch_size=batch_size)
            batches += 1
            if batches >= len(queries_train) / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

    vgg_netvlad.save_weights("model.h5")
    print("Saved model to disk")

    #plt.figure(figsize=(8, 8))
    #plt.plot(H.history['loss'], label='training loss')
    # plt.plot(H.history['val_loss'], label='validation loss')
    #plt.legend()
    #plt.title('Train/validation loss')
    #plt.show()

#%%

vgg_netvlad = my_model.get_netvlad_extractor()
vgg_netvlad.summary()
#result = vgg_netvlad.predict(images[:1])
#%%

print("Testing model")

############### test model
# this function create a perfect ranking :)
from sklearn.neighbors import NearestNeighbors

input_shape = (224, 224, 3)


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

#%%

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def create_image_dict(img_list):
    input_shape = (224, 224, 3)
    tensor = {}
    for path in img_list:
        img = image.load_img(path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img_key = path.strip('holidays_small/')
        tensor[img_key] = img
    # tensor = np.array(tensor)
    return tensor


#%%

img_dict = create_image_dict(get_imlist('holidays_small'))
img_dict.keys()

# img_tensor = images_to_tensor(imnames)
img_tensor = [img_dict[key] for key in img_dict]
img_tensor = np.array(img_tensor)

#%%
#vgg_netvlad.load_weights("model.h5")

#%%
# vgg_netvlad.summary()
all_feats = vgg_netvlad.predict(img_tensor)

#all_feats = all_feats[:, n_queries:]

plt.imshow(all_feats, cmap='viridis')
plt.colorbar()
plt.grid(False)
plt.show()

#%%

query_feats = all_feats[query_imids]

# SOLUTION
nbrs = NearestNeighbors(n_neighbors=1491, metric='cosine').fit(all_feats)
distances, indices = nbrs.kneighbors(query_feats)


#%%

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


#%%

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
        print(qno, (imfiles))
        plt.imshow(montage(imfiles, thumb_size=ts, ok=oks, shape=(1, nres)))
        plt.show()


#%%

show_result(indices, nqueries=50)
