import gc
import os
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from netvlad_model import NetVLADSiameseModel  # , NetVLADModelRetinaNet
from netvlad_model import input_shape
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import math
import paths

def get_imlist(path):
    return [f[:-len(".jpg")] for f in os.listdir(path) if f.endswith(".jpg")]


def main():
    print("Loading image dict")
    path_oxford = paths.path_oxford
    path_paris = paths.path_paris
    path_holidays = 'holidays_small/'

    format = 'buildings'
    if format is 'inria':
        path = path_holidays
    elif format is 'buildings':
        path = path_oxford
        # path = path_paris

    my_model = NetVLADSiameseModel()
    vgg_netvlad = my_model.build_netvladmodel()
    weight_name = "model_e424_adam-200-steps-per-epoch_.h5"

    print("Loading weights: " + weight_name)
    vgg_netvlad.load_weights(weight_name)
    vgg_netvlad = my_model.get_netvlad_extractor()

    base_resolution = (336, 336, 3)

    input_shape_1 = (768, 768, 3)
    input_shape_2 = (504, 504, 3)
    input_shape_3 = (224, 224, 3)
    input_shape_4 = (160, 160, 3)

    batch_size = 64
    input_shapes = [input_shape_1, input_shape_2]

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print("Loading images at shape: {}".format(base_resolution))
    gen = datagen.flow_from_directory(path, target_size=(base_resolution[0], base_resolution[1]), batch_size=batch_size,
                                      class_mode=None,
                                      shuffle=False, interpolation='bilinear')
    print("Computing descriptors")
    img_list = [os.path.splitext(os.path.split(f)[1])[0] for f in gen.filenames]
    n_steps = math.ceil(len(img_list)/batch_size)
    all_feats = vgg_netvlad.predict_generator(gen, steps=n_steps, verbose=1)

    multi_resolution = False
    if multi_resolution:
        for shape in input_shapes:
            print("Loading images at shape: {}".format(shape))
            gen = datagen.flow_from_directory(path, target_size=(shape[0], shape[1]),
                                              batch_size=batch_size,
                                              class_mode=None,
                                              shuffle=False, interpolation='bilinear')
            print("Computing descriptors")
            all_feats += vgg_netvlad.predict_generator(gen, steps=n_steps, verbose=1, use_multiprocessing=True)

    all_feats = normalize(all_feats)

    use_pca = False
    use_trained_pca = False

    if use_pca:
        if not use_trained_pca:
            print("Computing PCA")
            all_feats = PCA(512, svd_solver='full').fit_transform(all_feats)
        else:
            print("Loading PCA")
            pca = PCA()
            dataset = h5py.File("pca.h5", 'r')
            components = dataset['components'][:]
            mean = dataset['mean'][:]
            pca.components_ = components
            pca.mean_ = mean

            all_feats = pca.transform(all_feats)

    use_power_norm = True
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
