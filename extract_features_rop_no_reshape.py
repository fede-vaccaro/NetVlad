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
from sklearn.preprocessing import normalize
from torch.utils import data

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


def save_matrix(name, queries, database):
    dataset = h5py.File(name, 'w')
    dataset.create_dataset(name='queries', data=queries)
    dataset.create_dataset(name='database', data=database)
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


class ConfigDataset(data.Dataset):
    def __init__(self, cfg, transform, scale=1.0):
        self.cfg = cfg

        self.query = False
        self.transform = transform
        self.scale = scale

    def __len__(self):
        if self.query:
            return cfg['nq']
        else:
            return cfg['n']

    def __getitem__(self, i):
        if self.query:
            qim = pil_loader(self.cfg['qim_fname'](self.cfg, i)).crop(self.cfg['gnd'][i]['bbx'])
            # qim = make_square(qim)
            new_size = (qim.size[0]*self.scale, qim.size[1]*self.scale)
            new_size = (new_size[0].__int__(), new_size[1].__int__())
            qim = qim.resize(new_size, resample=Image.ANTIALIAS)

            qim = self.transform(qim)
            return qim
        else:
            im = pil_loader(self.cfg['im_fname'](self.cfg, i))
            new_size = (im.size[0]*self.scale, im.size[1]*self.scale)
            new_size = (new_size[0].__int__(), new_size[1].__int__())
            qim = im.resize(new_size, resample=Image.ANTIALIAS)

            im = self.transform(im)
            return im



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
    use_multi_resolution = True
    use_pca = True
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

    vladnet = nm.VLADNet(**network_conf)
    if torch.cuda.device_count() > 1:
        print("Available GPUS: ", torch.cuda.device_count())
        vladnet_parallel = torch.nn.DataParallel(vladnet, device_ids=range(torch.cuda.device_count()))

    # weight_name = "model_e300_resnet-101-torch-caffe-lrscheduling_0.9296_checkpoint.pkl"
    weight_name = model_name
    print("Loading weights: " + weight_name)
    checkpoint = torch.load(weight_name)
    vladnet_parallel.load_state_dict(checkpoint['model_state_dict'])
    vladnet_parallel.to('cuda')

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

    batch_size = 1
    config_dataset = ConfigDataset(cfg=cfg, transform=vladnet.get_transform())
    config_loader = data.DataLoader(dataset=config_dataset, num_workers=16, pin_memory=True, shuffle=False,
                                    batch_size=batch_size)

    query_scaling = 0.7
    resolutions = []

    if use_multi_resolution:
        resolutions += [np.sqrt(2), 1, 1/np.sqrt(2)]
    else:
        resolutions += [1]

    # resolutions = [r*side_res for r in resolutions]

    # extract queries
    print("Extracting query images")

    config_dataset.query = True

    Q_matrix = np.zeros((len(config_dataset), vladnet.output_dim))

    n_steps_queries = int(np.ceil(len(config_dataset) / batch_size))

    for res in resolutions:
        # res_ = int(res * query_scaling)
        # print("Extracting at resolution: {}".format(res_))
        # transform = vladnet.get_transform(res_)
        config_dataset.scale = res

        Q = utils.predict_generator_with_netlvad(model=vladnet_parallel, device='cuda', generator=config_loader,
                                                 n_steps=n_steps_queries, verbose=True)
        Q_matrix += Q

    Q_matrix = normalize(Q_matrix)

    if use_pca:
        Q_matrix = utils.transform(Q_matrix, mean=pca['mean'], components=pca['components'],
                                   explained_variance=pca['explained_variance'], whiten=True)

    Q_matrix = normalize(Q_matrix)

    # extract database images
    print("Extracting DB images")

    config_dataset.query = False

    DB_matrix = np.zeros((len(config_dataset), vladnet.output_dim))
    n_steps = int(np.ceil(len(config_dataset) / batch_size))

    for res in resolutions:
        # res_ = int(res) ro
        # print("Extracting at resolution: {}".format(res_))
        # transform = vladnet.get_transform(res_)
        # config_dataset.transform = transform
        config_dataset.scale = res

        DB_batch = utils.predict_generator_with_netlvad(model=vladnet_parallel, device='cuda', generator=config_loader,
                                                        n_steps=n_steps,
                                                        verbose=True)
        DB_matrix += DB_batch

    DB_matrix = normalize(DB_matrix)

    if use_pca:
        DB_matrix = utils.transform(DB_matrix, mean=pca['mean'], components=pca['components'],
                                    explained_variance=pca['explained_variance'], whiten=True)

    DB_matrix = normalize(DB_matrix)

    save_matrix('{}.h5'.format(test_dataset), Q_matrix, DB_matrix)
