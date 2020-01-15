import os
import subprocess
import sys

import h5py
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from netvlad_model import NetVLADSiameseModel  # , NetVLADModelRetinaNet
from netvlad_model import input_shape


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]


def create_image_dict(path):
    # input_shape = (224, 224, 3)
    img_list = get_imlist(path)
    tensor = {}
    for i, im_path in enumerate(img_list):
        if (i - 1) % 500 is 0:
            print("Img {} loaded".format(i))
        img = image.load_img(im_path, target_size=(input_shape[0], input_shape[1]), interpolation='bilinear')
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img_key = im_path[len(path):][:-len(".jpg")]
        # print(im_path, img_key)
        tensor[img_key] = img

    # tensor = np.array(tensor)
    return tensor


def main():
    print("Loading image dict")
    path_oxford = '/mnt/sdb-seagate/datasets/oxford5k/oxbuild_images_small/'
    path_paris = '/mnt/sdb-seagate/datasets/paris_small/'
    path_holidays = 'holidays_small/'

    format = 'buildings'
    if format is 'inria':
        path = path_holidays
    elif format is 'buildings':
        path = path_oxford
        path = path_paris

    img_dict = create_image_dict(path)

    # all_keys = [key for key in img_dict.keys()]
    all_keys = list(img_dict.keys())
    img_tensor = [img_dict[key] for key in img_dict.keys()]
    img_tensor = np.array(img_tensor)

    def sort_by_epoch(x):
        x = x[len("weights-netvlad-"):]
        x = x[:-len(".hdf5")]
        x = int(x)
        return x

    my_model = NetVLADSiameseModel()
    vgg_netvlad = my_model.build_netvladmodel()
    weight_name = "model_e111_sc-adam-2_0.0686.h5"

    print("Loading weights: " + weight_name)
    vgg_netvlad.load_weights(weight_name)
    vgg_netvlad = my_model.get_netvlad_extractor()

    #    for weight_name in sorted(os.listdir("/mnt/sdb-seagate/weights/"), key=sort_by_epoch):

    print("Computing descriptors")
    all_feats = vgg_netvlad.predict(img_tensor, verbose=1)

    del img_tensor

    # save all feats
    save_feat = False
    if save_feat:
        dataset = h5py.File("feats.h5", 'w')
        for feat, id in zip(all_feats, all_keys):
            dataset.create_dataset(id, data=feat)
        dataset.close()

    use_pca = True
    use_trained_pca = True

    if use_pca:
        if not use_trained_pca:
            print("Computing PCA")

            all_feats = PCA(512, svd_solver='full').fit_transform(all_feats)
            all_feats_sign = np.sign(all_feats)
            all_feats = np.power(np.abs(all_feats), 0.5)
            all_feats = np.multiply(all_feats, all_feats_sign)
        else:
            print("Loading PCA")
            pca = PCA()
            dataset = h5py.File("pca.h5", 'r')
            components = dataset['components'][:]
            mean = dataset['mean'][:]
            pca.components_ = components
            pca.mean_ = mean

            all_feats = pca.transform(all_feats)

    all_feats = normalize(all_feats)

    print("Computing NN")
    nbrs = NearestNeighbors(n_neighbors=len(all_keys), metric='cosine').fit(all_feats)

    # imnames = all_keys

    # query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]

    distances, indices = nbrs.kneighbors(all_feats)
    print(indices.shape)

    if format is 'buildings':
        for i, row in enumerate(indices):
            file = open("results/{}_ranked_list.dat".format(all_keys[i]), 'w')
            # file.write(all_keys[i] + " ")
            for j in row:
                file.write(all_keys[j] + "\n")
            # file.write("\n")
            file.close()

    elif format is 'inria':
        file = open("eval_holidays/holidays_results.dat", 'w')
        for i, row in enumerate(indices):
            if int(all_keys[i]) % 100 is 0:
                string = all_keys[i] + ".jpg "
                for k, j in enumerate(row[1:]):
                    string += "{} {}.jpg ".format(k, all_keys[j])
                file.write(string + "\n")
        file.close()

        sys.path.insert(1, '/path/to/application/app/folder')
        result = subprocess.check_output(
            'python2 ' + "eval_holidays/holidays_map.py " + "eval_holidays/holidays_results.dat",
            shell=True)
        print(result)


if __name__ == "__main__":
    main()
