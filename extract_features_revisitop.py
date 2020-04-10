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
import torch
import yaml
from PIL import Image, ImageFile

import netvlad_model as nm
import utils
from dataset import configdataset
from download import download_datasets


def get_imlist(path):
    return [f[:-len(".jpg")] for f in os.listdir(path) if f.endswith(".jpg")]


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


def save_dict_to_h5(name, dict):
    dataset = h5py.File(name, 'w')
    for key in dict.keys():
        desc = dict[key]
        dataset.create_dataset(name=str(key), data=desc[0])
    dataset.close()


def get_scaled_image_(img, target):
    x, y = img.size

    ratio = x / y
    if x > y:
        x_new = int(target * ratio)
        y_new = target
    else:
        x_new = target
        y_new = int(target / ratio)

    return x_new, y_new


def make_square(im, min_size=256, fill_color=(255, 255, 255)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def get_scaled_image(img, target):
    x, y = img.size

    r = x / y
    a = target ** 2

    y_new = int((a / r) ** 0.5)
    x_new = int(y_new * r)
    return x_new, y_new


def get_scaled_query(img, target_side):
    x, y = img.size
    ratio = x / y

    if x > y:
        new_x = target_side * ratio
        new_y = target_side
    else:
        new_x = target_side
        new_y = target_side / ratio

    return int(new_x), int(new_y)


def extract_feat(model, img, multiresolution=False, pca=None, query=False):
    with torch.no_grad():
        scaling_factor = 0.70
        #print(int(364 * scaling_factor))
        #print(int(546 * scaling_factor))
        #print(int(242 * scaling_factor))
        base_side = model.input_shape[0]
        img_1 = model.get_transform(int(base_side * scaling_factor) if query else base_side)(img).unsqueeze(0)
        desc = model.predict_with_netvlad(img_1)

        if multiresolution:
            img_2 = model.get_transform(int(base_side * 3/2 * scaling_factor) if query else int(base_side * 3/2))(img).unsqueeze(0)
            desc += model.predict_with_netvlad(img_2)

            img_3 = model.get_transform(int(base_side * 2/3 * scaling_factor) if query else int(base_side * 2/3))(img).unsqueeze(0)
            desc += model.predict_with_netvlad(img_3)

            desc /= np.linalg.norm(desc, ord=2)

        desc_local = desc

        if pca is not None:
            desc = utils.transform(desc, mean=pca["mean"], components=pca["components"],
                                   explained_variance=pca["explained_variance"], whiten=True,
                                   pow_whiten=0.5)

        desc /= np.linalg.norm(desc, ord=2)

        return desc


if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--model", type=str,
                    help="path to *specific* model checkpoint to load")
    ap.add_argument("-c", "--configuration", type=str, default='resnet-conf.yaml',
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
    paris_dataset = args['paris']
    oxford_dataset = args['oxford']

    conf_file = open(config_file, 'r')
    conf = dict(yaml.safe_load(conf_file))
    conf_file.close()

    use_power_norm = conf['use_power_norm']
    use_multi_resolution = conf['use_multi_resolution']
    side_res = conf['input-shape']

    nm.NetVladBase.input_shape = (side_res, side_res, 3)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # ---------------------------------------------------------------------
    # Set data folder and testing parameters
    # ---------------------------------------------------------------------
    # Set data folder, change if you have downloaded the data somewhere else
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    # Check, and, if necessary, download test data (Oxford and Pairs) and revisited annotation
    download_datasets(data_root)

    # Set test dataset: roxford5k | rparis6k
    test_dataset = 'roxford5k' if oxford_dataset else 'rparis6k'

    print('>> {}: Processing test dataset...'.format(test_dataset))
    # config file for the dataset
    # separates query image list from database image list, if revisited protocol used
    cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))


    print("Loading image dict")

    network_conf = conf['network']
    net_name = network_conf['name']

    vladnet = None
    if net_name == "vgg":
        vladnet = nm.NetVLADSiameseModel(**network_conf)
    elif net_name == "resnet":
        vladnet = nm.NetVladResnet(**network_conf)
        # vladnet = nm.GeMResnet(**network_conf)
    else:
        print("Network name not valid.")

    # weight_name = "model_e300_resnet-101-torch-caffe-lrscheduling_0.9296_checkpoint.pkl"
    weight_name = model_name
    print("Loading weights: " + weight_name)
    checkpoint = torch.load(weight_name)
    vladnet.load_state_dict(checkpoint['model_state_dict'])
    vladnet.cuda()
    #

    # ---------------------------------------------------------------------
    # Set data folder and testing parameters
    # ---------------------------------------------------------------------
    # Set data folder, change if you have downloaded the data somewhere else
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    # Check, and, if necessary, download test data (Oxford and Pairs) and revisited annotation
    download_datasets(data_root)

    pca_dataset = h5py.File("pca_{}.h5".format(weight_name.split("/")[-1]), 'r')
    mean = pca_dataset['mean'][:]
    components = pca_dataset['components'][:]
    explained_variance = pca_dataset['explained_variance'][:]
    pca_dataset.close()

    pca = {}
    pca['mean'] = mean
    pca['components'] = components
    pca['explained_variance'] = explained_variance

    Q = {}

    # query images
    for i in np.arange(cfg['nq']):
        qim = pil_loader(cfg['qim_fname'](cfg, i)).crop(cfg['gnd'][i]['bbx'])
        # qim.show()
        # time.sleep(1)
        qim = make_square(qim)
        Q[str(i)] = extract_feat(vladnet, qim, multiresolution=True, pca=pca, query=True)

        print('>> {}: Processing query image {}'.format(test_dataset, i + 1))
    save_dict_to_h5('Q_{}.h5'.format(test_dataset), Q)

    X = {}

    for i in np.arange(cfg['n']):
        im = pil_loader(cfg['im_fname'](cfg, i))
        X[str(i)] = extract_feat(vladnet, im, multiresolution=True, pca=pca)

        print('>> {}: Processing database image {}'.format(test_dataset, i + 1))

    save_dict_to_h5('X_{}.h5'.format(test_dataset), X)
