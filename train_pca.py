import argparse
import os
import utils
import h5py
import torch
import yaml
from sklearn.decomposition import PCA
from torchvision.datasets import folder

import netvlad_model as nm
import paths
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-m", "--model", type=str,
                    help="path to *specific* model checkpoint to load")
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
    if False:
        nm.NetVladBase.input_shape = (None, None, 3)


    def get_imlist(path):
        return [f[:-len(".jpg")] for f in os.listdir(path) if f.endswith(".jpg")]


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    network_conf = conf['network']
    # net_name = network_conf['name']

    vladnet = nm.VLADNet(**network_conf)

    transform = vladnet.full_transform
    device = "cuda"

    if torch.cuda.device_count() > 1:
        print("Available GPUS: ", torch.cuda.device_count())
        vladnet = torch.nn.DataParallel(vladnet, device_ids=range(torch.cuda.device_count()))


    weight_name = model_name
    print("Loading weights: " + weight_name)
    checkpoint = torch.load(model_name)
    vladnet.load_state_dict(checkpoint['model_state_dict'])
    vladnet.to(device)

    print("PCA is going to be saved to: ", "pca_{}.h5".format(weight_name.split('/')[-1]))

    train_pca(vladnet, "pca_{}.h5".format(weight_name.split('/')[-1]), transform)


def train_pca(vladnet, out_name, transform):
    image_folder = folder.ImageFolder(root=paths.landmarks_path, transform=transform)
    gen = torch.utils.data.DataLoader(
        image_folder,
        batch_size=16,
        num_workers=32,
        shuffle=True,
    )
    all_feats = utils.predict_generator_with_netlvad(device='cuda', generator=gen, n_steps=4096, model=vladnet)
    print("All descs shape: ", all_feats.shape)
    print("Sampling local features")
    print("Computing PCA")
    dim_pca = 2048
    pca = PCA(dim_pca)
    pca.fit(all_feats)
    pca_dataset = h5py.File(out_name, 'w')
    pca_dataset.create_dataset('components', data=pca.components_)
    pca_dataset.create_dataset('mean', data=pca.mean_)
    pca_dataset.create_dataset('explained_variance', data=pca.explained_variance_)
    pca_dataset.close()


if __name__ == "__main__":
    main()
