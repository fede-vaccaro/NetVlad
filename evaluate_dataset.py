import gc
import os
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image
from keras import Model
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
from triplet_loss import TripletL2LossLayerSoftmax

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
        # my_model = nm.NetVladResnet(**network_conf)
        my_model = nm.GeMResnet(**network_conf)
    else:
        print("Network name not valid.")
        quit()

    vgg_netvlad = my_model.build_netvladmodel()
    triplet_loss_layer = TripletL2LossLayerSoftmax(n_classes=28744, alpha=0.1, l=0.5)
    vgg_netvlad = Model(vgg_netvlad.input, triplet_loss_layer(vgg_netvlad.output))
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
        APs = []

        queries = {}
        # open queries

        if test_oxford:
            gt_path = "gt-oxford"
        else:
            gt_path = "gt-paris"

        text_files = os.listdir(gt_path)

        for file in text_files:
            if file.endswith("_query.txt"):
                query_file = open(gt_path + "/" + file, 'r')

                if test_oxford:
                    query_pic = query_file.readline().split(" ")[0][len("oxc1_"):]
                else:
                    query_pic = query_file.readline().split(" ")[0]

                query_name = file[:-len("_query.txt")]
                queries[query_pic] = query_name

        for row in indices:
            file_name = img_list[row[0]]
            if file_name in set(queries.keys()):
                ranked_list = []
                for j in row:
                    ranked_list += [img_list[j]]
                ap = compute_ap(query_name=queries[file_name], ranked_list=ranked_list, gt_path=gt_path)
                APs.append(ap)

        print("mAP is: {}".format(np.array(APs).mean()))

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

def load_set(set_name):
    files = []

    with open(set_name, 'r') as text:
        for line in text.readlines():
            name = line.rstrip("\n")
            files.append(name)

    return set(files)

def compute_ap(query_name, ranked_list, gt_path):
    good_set = load_set(gt_path + "/" + query_name + "_good.txt")
    ok_set = load_set(gt_path + "/" + query_name + "_ok.txt")
    junk_set = load_set(gt_path + "/" + query_name + "_junk.txt")

    pos_set = good_set.union(ok_set)

    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0

    intersect_size = 0

    j = 0
    for i, el in enumerate(ranked_list):
        if el in junk_set:
            continue
        if el in pos_set:
            intersect_size += 1

        recall = intersect_size / len(pos_set)
        precision = intersect_size / (j + 1.0)

        ap += (recall - old_recall)*((old_precision + precision)/2.0)
        old_recall = recall
        old_precision = precision
        j += 1

    return ap



if __name__ == "__main__":
    main()
