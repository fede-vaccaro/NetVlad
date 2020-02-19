import gc
import os
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import netvlad_model as nm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import math
import paths
import argparse
import yaml
import utils
ap = argparse.ArgumentParser()


ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-c", "--configuration", type=str, default='train_configuration.yaml',
                help="Yaml file where the configuration is stored")
ap.add_argument("-d", "--device", type=str, default="0",
                help="CUDA device to be used. For info type '$ nvidia-smi'")
ap.add_argument("-p", "--paris", action='store_true',
                help="Test Paris6K")
ap.add_argument("-o", "--oxford", action='store_true',
                help="Test Oxford5K")

args = vars(ap.parse_args())

model_name = args['model']
config_file = args['configuration']
cuda_device = args['device']

test_paris = args['paris']
test_oxford = args['oxford']

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


def main():
    print("Loading image dict")
    path_oxford = paths.path_oxford
    path_paris = paths.path_paris
    path_holidays = 'holidays_2/'

    format = 'buildings'
    if format is 'inria':
        path = path_holidays
    elif format is 'buildings':
        if test_oxford:
            path = path_oxford
        elif test_paris:
            path = path_paris

    network_conf = conf['network']
    net_name = network_conf['name']

    my_model = None
    if net_name == "vgg":
        my_model = nm.NetVLADSiameseModel(**network_conf)
    elif net_name == "resnet":
        my_model = nm.NetVladResnet(**network_conf)
        # my_model = nm.GeMResnet(**network_conf)
    else:
        print("Network name not valid.")

    vgg_netvlad = my_model.build_netvladmodel()
    weight_name = model_name
    print("Loading weights: " + weight_name)
    vgg_netvlad.load_weights(weight_name)
    vgg_netvlad = my_model.get_netvlad_extractor()

    base_resolution = (side_res, side_res, 3)

    input_shape_1 = (768, 768, 3)
    input_shape_2 = (504, 504, 3)
    input_shape_3 = (224, 224, 3)
    input_shape_4 = (160, 160, 3)

    batch_size = 16
    input_shapes = [input_shape_2, input_shape_3]

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print("Loading images at shape: {}".format(base_resolution))
    gen = datagen.flow_from_directory(path, target_size=(base_resolution[0], base_resolution[1]), batch_size=batch_size,
                                      class_mode=None,
                                      shuffle=False, interpolation='bilinear')
    print("Computing descriptors")
    img_list = [os.path.splitext(os.path.split(f)[1])[0] for f in gen.filenames]
    n_steps = math.ceil(len(img_list) / batch_size)
    all_feats = vgg_netvlad.predict_generator(gen, steps=n_steps, verbose=1)

    if use_multi_resolution:
        for shape in input_shapes:
            print("Loading images at shape: {}".format(shape))
            gen = datagen.flow_from_directory(path, target_size=(shape[0], shape[1]),
                                              batch_size=batch_size,
                                              class_mode=None,
                                              shuffle=False, interpolation='bilinear')
            print("Computing descriptors")
            all_feats += vgg_netvlad.predict_generator(gen, steps=n_steps, verbose=1)

    use_pca = False
    if use_pca:
        n_components = 4096

        pca_dataset = h5py.File("pca_{}.h5".format(n_components), 'r')
        mean = pca_dataset['mean'][:]
        components = pca_dataset['components'][:]
        explained_variance = pca_dataset['explained_variance'][:]
        pca_dataset.close()

        all_feats = utils.transform(all_feats, mean, components, explained_variance, whiten=True, pow_whiten=0.5)

    all_feats = normalize(all_feats)

    if use_power_norm:
        all_feats_sign = np.sign(all_feats)
        all_feats = np.power(np.abs(all_feats), 0.5)
        all_feats = np.multiply(all_feats, all_feats_sign)

    all_feats = normalize(all_feats)

    print("Computing NN")
    nbrs = NearestNeighbors(n_neighbors=len(img_list), metric='cosine').fit(all_feats)

    # imnames = all_keys

    # query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]

    distances, indices = nbrs.kneighbors(all_feats)
    print(indices.shape)

    if format is 'buildings':
        for i, row in tqdm(enumerate(indices)):
            file = open("results/{}_ranked_list.dat".format(img_list[i]), 'w')
            # file.write(all_keys[i] + " ")
            for j in row:
                file.write(img_list[j] + "\n")
            # file.write("\n")
            file.close()

    elif format is 'inria':
        file = open("eval_holidays/holidays_results.dat", 'w')
        for i, row in enumerate(indices):
            if int(img_list[i]) % 100 is 0:
                string = img_list[i] + ".jpg "
                for k, j in enumerate(row[1:]):
                    string += "{} {}.jpg ".format(k, img_list[j])
                file.write(string + "\n")
        file.close()

        sys.path.insert(1, '/path/to/application/app/folder')
        result = subprocess.check_output(
            'python2 ' + "eval_holidays/holidays_map.py " + "eval_holidays/holidays_results.dat",
            shell=True)
        print(result)


if __name__ == "__main__":
    main()
