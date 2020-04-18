# EXAMPLE_PROCESS_IMAGES  Code to read and process images for ROxford and RParis datasets.
# Revisited protocol requires query images to be removed from the database, and cropped prior to any processing.
# This code makes sure the protocol is strictly followed.
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018

import argparse
import os

import h5py
import numpy as np
import yaml
from PIL import Image, ImageFile

import netvlad_model as nm
import utils
from dataset import configdataset
from download import download_datasets
from keras.applications.vgg16 import preprocess_input

# ---------------------------------------------------------------------
# Set data folder and testing parameters
# ---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
# Check, and, if necessary, download test data (Oxford and Pairs) and revisited annotation
download_datasets(data_root)

# Set test dataset: roxford5k | rparis6k
test_dataset = 'rparis6k'#roxford5k'


# ---------------------------------------------------------------------
# Read images
# ---------------------------------------------------------------------

def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


print('>> {}: Processing test dataset...'.format(test_dataset))
# config file for the dataset
# separates query image list from database image list, if revisited protocol used
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load",
                default="resnet101_model_e373_resnet-101_0.9362_checkpoint.h5")
ap.add_argument("-c", "--configuration", type=str, default='resnet-conf.yaml',
                help="Yaml file where the configuration is stored")
ap.add_argument("-d", "--device", type=str, default="0",
                help="CUDA device to be used. For info type '$ nvidia-smi'")

args = vars(ap.parse_args())

model_name = args['model']
config_file = args['configuration']
cuda_device = args['device']

conf_file = open(config_file, 'r')
conf = dict(yaml.safe_load(conf_file))
conf_file.close()

use_power_norm = conf['use_power_norm']
use_multi_resolution = conf['use_multi_resolution']
side_res = conf['input-shape']

nm.NetVladBase.input_shape = (side_res, side_res, 3)
if use_multi_resolution:
    nm.NetVladBase.input_shape = (None, None, 3)


def get_imlist(path):
    return [f[:-len(".jpg")] for f in os.listdir(path) if f.endswith(".jpg")]


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

print("Loading image dict")

network_conf = conf['network']
net_name = network_conf['name']

my_model = None
if net_name == "vgg":
    my_model = nm.NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    my_model = nm.VLADNet(**network_conf)
    # my_model = nm.GeMResnet(**network_conf)
else:
    print("Network name not valid.")

vgg_netvlad = my_model.build_netvladmodel()
weight_name = model_name
print("Loading weights: " + weight_name)
vgg_netvlad.load_weights(weight_name)
vgg_netvlad = my_model.get_netvlad_extractor()

def preprocess(im):
    im = np.array(im)
    return np.expand_dims(preprocess_input(im), 0)


def resize(im, res):
    return im.resize((res, res), Image.BILINEAR)

def extract_image(im, model, pca):
    im_0 = preprocess(resize(im, 504))
    feat = model.predict(im_0)

    im_1 = preprocess(resize(im, 336))
    feat += model.predict(im_1)

    im_2 = preprocess(resize(im, 224))
    feat += model.predict(im_2)

    feat /= np.linalg.norm(feat, ord=2)

    feat = utils.transform(feat, components=pca['components'], mean=pca['mean'], explained_variance=pca['explained_variance'],
                    whiten=True)

    feat /= np.linalg.norm(feat, ord=2)

    return feat


pca_dataset = h5py.File("pca_resnet101_2048.h5", 'r')
mean = pca_dataset['mean'][:]
components = pca_dataset['components'][:]
explained_variance = pca_dataset['explained_variance'][:]
pca_dataset.close()

pca = {'components': components, 'mean':mean, 'explained_variance':explained_variance}

queries = h5py.File("Q.h5", 'w')

# query images
for i in np.arange(cfg['nq']):
    qim = pil_loader(cfg['qim_fname'](cfg, i))#.crop(cfg['gnd'][i]['bbx'])

    feat = extract_image(qim, vgg_netvlad, pca)

    queries.create_dataset(str(i), data=feat[0])

    print('>> {}: Processing query image {}'.format(test_dataset, i + 1))

queries.close()

images = h5py.File("X.h5", 'w')
for i in np.arange(cfg['n']):
    im = pil_loader(cfg['im_fname'](cfg, i))
    feat = extract_image(im, vgg_netvlad, pca)

    images.create_dataset(str(i), data=feat[0])
    print('>> {}: Processing database image {}'.format(test_dataset, i + 1))
images.close()