# %%

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import os
import numpy as np


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


# %%

img_dict = create_image_dict(get_imlist('holidays_small'))
img_dict.keys()

# %%

# create dataset
queries = []
positives = []
negatives = []

triplets_file = open("triplets.dat", "r")
for line in triplets_file.readlines():
    split = line.split(" ")[:3]

    queries.append(img_dict[split[0]])
    positives.append(img_dict[split[1]])
    negatives.append(img_dict[split[2]])

queries = np.array(queries)
positives = np.array(positives)
negatives = np.array(negatives)

# %%

# create model
from keras.applications import VGG16

embedding_size = 512

input_shape = (224, 224, 3)

vgg = VGG16(weights='imagenet', include_top=False, pooling=False, input_shape=input_shape)

for layer in vgg.layers:
    layer.trainable = False
    # print(layer, layer.trainable)

vgg.layers.pop(0)
vgg.summary()

# %%

# set layers untrainable
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Permute
from keras.optimizers import Adam
from keras.utils import plot_model
from triplet_loss import L2NormLayer, Transpose
from loupe_keras import NetVLAD

input_q = Input(shape=(224, 224, 3))
input_p = Input(shape=(224, 224, 3))
input_n = Input(shape=(224, 224, 3))

vgg_q = vgg(input_q)
vgg_p = vgg(input_p)
vgg_n = vgg(input_n)

resnet_output = vgg.output_shape[1]

transpose = Permute((3, 1, 2), input_shape=(7, 7, 512))
reshape = Reshape((2048, 7 * 7))
netvlad = NetVLAD(feature_size=7 * 7, max_samples=512, cluster_size=64)#, output_dim=1024)
"""l2normalization = L2NormLayer()
dropout = Dropout(0.1)
"""

# %%

embedding_q = netvlad(reshape(transpose(vgg_q)))
embedding_p = netvlad(reshape(transpose(vgg_p)))
embedding_n = netvlad(reshape(transpose(vgg_n)))

vgg_qpn = Model([input_q, input_p, input_n], [embedding_q, embedding_p, embedding_n])

# %%

plot_model(vgg_qpn, to_file='base_network.png', show_shapes=True, show_layer_names=True)
vgg_qpn.summary()

# %%

result = vgg_qpn.predict([queries[:1], positives[:1], negatives[:1]])

# %%

all_data_len = len(img_dict.keys())
n_train = 1200

fake_true_pred = np.zeros((n_train, embedding_size * 3))
fake_true_pred_val = np.zeros((all_data_len - n_train, embedding_size * 3))

queries_train = queries[:n_train]
positives_train = positives[:n_train]
negatives_train = negatives[:n_train]

queries_test = queries[n_train:]
positives_test = positives[n_train:]
negatives_test = negatives[n_train:]

# %%

from triplet_loss import TripletLossLayer

batch_size = 128
epochs = 32

# train session
opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!

loss_layer = TripletLossLayer(alpha=.1, name='triplet_loss_layer')(vgg_qpn.output)
vgg_qpn = Model(inputs=vgg_qpn.input, outputs=loss_layer)
vgg_qpn.compile(optimizer=opt)

# %%

H = vgg_qpn.fit(
    x=[queries_train, positives_train, negatives_train],
    y=None,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([queries_test, positives_test, negatives_test], None),
    verbose=1,
)

import matplotlib.pyplot as plt

vgg_qpn.save_weights("model.h5")
print("Saved model to disk")

plt.figure(figsize=(8, 8))
plt.plot(H.history['loss'], label='training loss')
plt.plot(H.history['val_loss'], label='validation loss')
plt.legend()
plt.title('Train/validation loss')
plt.show()

# %%

# pop triplet loss layer
# resnet_qpn_no_loss = Model(input=resnet_qpn.input, outputs=resnet_qpn.output)
# resnet_qpn_no_loss.layers.pop()
# resnet_qpn_no_loss.summary()


# %%

# reload model from disk
vgg_qpn = Model([input_q, input_p, input_n], [embedding_q, embedding_p, embedding_n])

vgg_qpn.load_weights('model.h5')

result = vgg_qpn.predict([queries[:1], positives[:1], negatives[:1]])
# %%

# test model

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

# %%

# img_tensor = images_to_tensor(imnames)
img_tensor = [img_dict[key] for key in img_dict]
img_tensor = np.array(img_tensor)

# %%
vgg_qpn.summary()
all_feats, _, _ = vgg_qpn.predict([img_tensor, np.zeros(img_tensor.shape), np.zeros(img_tensor.shape)])

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
