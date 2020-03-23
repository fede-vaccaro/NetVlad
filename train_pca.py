import argparse
import os

import h5py
import torch
import yaml
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.decomposition import PCA
from torchvision.datasets import folder

import netvlad_model as nm
import paths

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-c", "--configuration", type=str, default='train_configuration.yaml',
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

network_conf = conf['network']
net_name = network_conf['name']

vladnet = None
if net_name == "vgg":
    vladnet = nm.NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    vladnet = nm.NetVladResnet(**network_conf)
else:
    print("Network name not valid.")

weight_name = model_name
print("Loading weights: " + weight_name)
checkpoint = torch.load(model_name)
vladnet.load_state_dict(checkpoint['model_state_dict'])
vladnet.cuda()

image_folder = folder.ImageFolder(root=paths.landmarks_path, transform=vladnet.full_transform)
gen = torch.utils.data.DataLoader(
    image_folder,
    batch_size=16,
    num_workers=8,
    shuffle=True,
)


all_feats = vladnet.predict_generator_with_netlvad(generator=gen, n_steps=4096)

print("All descs shape: ", all_feats.shape)

print("Sampling local features")

print("Computing PCA")
dim_pca = 2048
pca = PCA(dim_pca)

pca.fit(all_feats)

pca_dataset = h5py.File("pca_{}.h5".format(dim_pca), 'w')
pca_dataset.create_dataset('components', data=pca.components_)
pca_dataset.create_dataset('mean', data=pca.mean_)
pca_dataset.create_dataset('explained_variance', data=pca.explained_variance_)
pca_dataset.close()
