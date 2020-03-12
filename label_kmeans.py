import math
import os
import shutil
import threading

import matplotlib.pyplot as plt
import numpy as np
import yaml
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

import netvlad_model as nm

dest_path = "/mnt/sdb-seagate/datasets/dataset_clustered/"
path_dataset = '/mnt/m2/dataset_compact/'


def restore(x, data_format='channels_first'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if not data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return (x - 127.5) / 255.0


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    # plt.show()


if not os.path.exists(dest_path):
    os.mkdir(dest_path)

conf_file = open('best-model-conf.yaml', 'r')
conf = dict(yaml.safe_load(conf_file))
conf_file.close()

use_power_norm = conf['use_power_norm']
use_multi_resolution = conf['use_multi_resolution']
side_res = conf['input-shape']

network_conf = conf['network']
net_name = network_conf['name']

nm.NetVladBase.input_shape = (side_res, side_res, 3)
if use_multi_resolution:
    nm.NetVladBase.input_shape = (None, None, 3)

my_model = None
if net_name == "vgg":
    my_model = nm.NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    my_model = nm.NetVladResnet(**network_conf)
else:
    print("Network name not valid.")

vgg_netvlad = my_model.build_netvladmodel()
weight_name = "best_model.h5"
print("Loading weights: " + weight_name)
vgg_netvlad.load_weights(weight_name)
vgg_netvlad = my_model.get_netvlad_extractor()

from keras.applications import ResNet50

vgg_netvlad = ResNet50(input_shape=(336, 336, 3), include_top=False, pooling='avg', weights='imagenet')


# vgg_netvlad = models.Model(vgg_netvlad.get_input_at(0), layers.GlobalAveragePooling2D()(vgg_netvlad.get_layer('block5_conv2').output))

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.txt')]


def open_img(path, input_shape=nm.NetVladBase.input_shape):
    img = image.load_img(path, target_size=(input_shape[0], input_shape[1]), interpolation='bilinear')
    # img = (image.img_to_array(img) - 127.5) / 127.5
    img = preprocess_input(image.img_to_array(img))
    img_id = path.split('/')[-1]

    return img, img_id


dirs = os.listdir(path_dataset)


# random.shuffle(dirs)

for label in tqdm(dirs):
    path_dataset_ = os.path.join(path_dataset, label)
    imgs = [os.path.join(path_dataset_, img) for img in os.listdir(path_dataset_)]
    img_tensor = np.asarray([x for x, _ in [open_img(p) for p in imgs]])
    img_descs = vgg_netvlad.predict(img_tensor)
    img_descs = normalize(img_descs)
    # img_descs = PCA(21).fit_transform(img_descs)

    clust_iters = range(2, 10)

    ss = []
    elbow = []

    kmeanss = []

    for i in clust_iters:
        kmeans = MiniBatchKMeans(n_clusters=i).fit(img_descs)
        sscore = silhouette_score(img_descs, kmeans.labels_, metric='euclidean')
        elbow.append(np.exp(kmeans.inertia_))
        ss.append(sscore)
        kmeanss.append(kmeans)

    # plt.plot(ss)
    # plt.show()

    best_kmeans_id = int(np.argmax(ss))
    # if best_kmeans_id is 0:
    #     best_kmeans = MiniBatchKMeans(n_clusters=1).fit(img_descs)
    # else:
    best_kmeans = kmeanss[best_kmeans_id]
    predictions = best_kmeans.predict(img_descs)

    n_cluster = best_kmeans.n_clusters
    clusters_img = [list() for l in range(n_cluster)]
    clusters_path = [list() for l in range(n_cluster)]

    for i, p in enumerate(predictions):
        clust_id = int(p)
        clusters_img[clust_id].append(img_tensor[i] * 0.5 + 0.5)
        clusters_path[clust_id].append(imgs[i])

    # copies images to a new dir
    bigger_cluster = max(clusters_path, key=lambda x: len(x))
    bigger_cluster_id = clusters_path.index(bigger_cluster)

    dest_dir = os.path.join(dest_path, label)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    for pic in bigger_cluster:
        im_path = os.path.join(dest_dir, pic.split("/")[-1])
        shutil.copy(pic, im_path)

import math
import os
import shutil
import threading

import matplotlib.pyplot as plt
import numpy as np
import yaml
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

import netvlad_model as nm

dest_path = "/mnt/sdb-seagate/datasets/dataset_clustered/"
path_dataset = '/mnt/m2/dataset_compact/'


def restore(x, data_format='channels_first'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if not data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return (x - 127.5) / 255.0


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    # plt.show()


if not os.path.exists(dest_path):
    os.mkdir(dest_path)

conf_file = open('best-model-conf.yaml', 'r')
conf = dict(yaml.safe_load(conf_file))
conf_file.close()

use_power_norm = conf['use_power_norm']
use_multi_resolution = conf['use_multi_resolution']
side_res = conf['input-shape']

network_conf = conf['network']
net_name = network_conf['name']

nm.NetVladBase.input_shape = (side_res, side_res, 3)
if use_multi_resolution:
    nm.NetVladBase.input_shape = (None, None, 3)

my_model = None
if net_name == "vgg":
    my_model = nm.NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    my_model = nm.NetVladResnet(**network_conf)
else:
    print("Network name not valid.")

vgg_netvlad = my_model.build_netvladmodel()
weight_name = "best_model.h5"
print("Loading weights: " + weight_name)
vgg_netvlad.load_weights(weight_name)
vgg_netvlad = my_model.get_netvlad_extractor()

from keras.applications import ResNet50

vgg_netvlad = ResNet50(input_shape=(336, 336, 3), include_top=False, pooling='avg', weights='imagenet')


# vgg_netvlad = models.Model(vgg_netvlad.get_input_at(0), layers.GlobalAveragePooling2D()(vgg_netvlad.get_layer('block5_conv2').output))

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.txt')]


def open_img(path, input_shape=nm.NetVladBase.input_shape):
    img = image.load_img(path, target_size=(input_shape[0], input_shape[1]), interpolation='bilinear')
    # img = (image.img_to_array(img) - 127.5) / 127.5
    img = preprocess_input(image.img_to_array(img))
    img_id = path.split('/')[-1]

    return img, img_id


dirs = os.listdir(path_dataset)


# random.shuffle(dirs)

for label in tqdm(dirs):
    path_dataset_ = os.path.join(path_dataset, label)
    imgs = [os.path.join(path_dataset_, img) for img in os.listdir(path_dataset_)]
    img_tensor = np.asarray([x for x, _ in [open_img(p) for p in imgs]])
    img_descs = vgg_netvlad.predict(img_tensor)
    img_descs = normalize(img_descs)
    # img_descs = PCA(21).fit_transform(img_descs)

    clust_iters = range(2, 10)

    ss = []
    elbow = []

    kmeanss = []

    for i in clust_iters:
        kmeans = MiniBatchKMeans(n_clusters=i).fit(img_descs)
        sscore = silhouette_score(img_descs, kmeans.labels_, metric='euclidean')
        elbow.append(np.exp(kmeans.inertia_))
        ss.append(sscore)
        kmeanss.append(kmeans)

    # plt.plot(ss)
    # plt.show()

    best_kmeans_id = int(np.argmax(ss))
    # if best_kmeans_id is 0:
    #     best_kmeans = MiniBatchKMeans(n_clusters=1).fit(img_descs)
    # else:
    best_kmeans = kmeanss[best_kmeans_id]
    predictions = best_kmeans.predict(img_descs)

    n_cluster = best_kmeans.n_clusters
    clusters_img = [list() for l in range(n_cluster)]
    clusters_path = [list() for l in range(n_cluster)]

    for i, p in enumerate(predictions):
        clust_id = int(p)
        clusters_img[clust_id].append(img_tensor[i] * 0.5 + 0.5)
        clusters_path[clust_id].append(imgs[i])

    # copies images to a new dir
    bigger_cluster = max(clusters_path, key=lambda x: len(x))
    bigger_cluster_id = clusters_path.index(bigger_cluster)

    dest_dir = os.path.join(dest_path, label)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    for pic in bigger_cluster:
        im_path = os.path.join(dest_dir, pic.split("/")[-1])
        shutil.copy(pic, im_path)

