import argparse
import os

import h5py
import yaml
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.decomposition import PCA

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

my_model = None
if net_name == "vgg":
    my_model = nm.NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    my_model = nm.NetVladResnet(**network_conf)
else:
    print("Network name not valid.")

vgg_netvlad = my_model.build_netvladmodel()
weight_name = model_name
print("Loading weights: " + weight_name)
vgg_netvlad.load_weights(weight_name)
vgg_netvlad = my_model.get_netvlad_extractor()

kmeans_generator = image.ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    paths.landmarks_path,
    target_size=(nm.NetVladBase.input_shape[0], nm.NetVladBase.input_shape[1]),
    batch_size=64//4,
    class_mode=None,
    interpolation='bilinear', seed=4242)

all_descs = vgg_netvlad.predict_generator(generator=kmeans_generator, steps=int(1024)*2, verbose=1)
print("All descs shape: ", all_descs.shape)

print("Sampling local features")

print("Computing PCA")
dim_pca = 2048
pca = PCA(dim_pca)

pca.fit(all_descs)

pca_dataset = h5py.File("pca_{}.h5".format(dim_pca), 'w')
pca_dataset.create_dataset('components', data=pca.components_)
pca_dataset.create_dataset('mean', data=pca.mean_)
pca_dataset.create_dataset('explained_variance', data=pca.explained_variance_)
pca_dataset.close()
