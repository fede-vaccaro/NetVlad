import argparse
import math
import os

import h5py
import numpy as np
import torch
import yaml
from sklearn.preprocessing import normalize
from torchvision.datasets import folder

import netvlad_model as nm
import open_dataset_utils as my_utils
import paths
import utils
from train_pca import train_pca


def main():
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
    ap.add_argument("-w", "--whitening", action='store_true',
                    help="Use PCA/Whitening")
    ap.add_argument("-mr", "--multiresolution", action='store_true',
                    help="Use MultiResolution descriptor")
    args = vars(ap.parse_args())

    model_name = args['model']
    config_file = args['configuration']
    cuda_device = args['device']

    test_paris = args['paris']
    test_oxford = args['oxford']

    use_pca = args['whitening']
    use_multi_resolution = args['multiresolution']

    conf_file = open(config_file, 'r')
    conf = dict(yaml.safe_load(conf_file))
    conf_file.close()

    use_power_norm = conf['use_power_norm']
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

    vladnet = None
    if net_name == "vgg":
        vladnet = nm.NetVLADSiameseModel(**network_conf)
    elif net_name == "resnet":
        vladnet = nm.NetVladResnet(**network_conf)
        # vladnet = nm.GeMResnet(**network_conf)
    else:
        print("Network name not valid.")

    weight_name = model_name
    print("Loading weights: " + weight_name)
    checkpoint = torch.load(model_name)
    vladnet.load_state_dict(checkpoint['model_state_dict'])
    vladnet.cuda()

    dataset = ""
    if test_oxford:
        dataset = 'o'
    elif test_paris:
        dataset = 'p'

    APs = compute_aps(model=vladnet, dataset=dataset, use_power_norm=use_power_norm,
                      use_multi_resolution=use_multi_resolution, base_resolution=side_res, verbose=True,
                      pca=None if not use_pca else "pca_{}.h5".format(weight_name))

    print("mAP is: {}".format(np.array(APs).mean()))


def compute_aps(model, dataset='o', use_power_norm=False, use_multi_resolution=False, base_resolution=336,
                verbose=False, pca=None):
    path_oxford = paths.path_oxford
    path_paris = paths.path_paris
    if dataset == 'o':
        dataset_path = path_oxford
    else:
        dataset_path = path_paris

    base_resolution = model.input_shape
    input_shape_1 = (768, 768, 3)
    input_shape_2 = (504, 504, 3)
    input_shape_3 = (224, 224, 3)
    input_shape_4 = (160, 160, 3)
    batch_size = 16
    input_shapes = [input_shape_2, input_shape_3]

    print("Loading images at shape: {}".format(base_resolution))
    image_folder = folder.ImageFolder(root=dataset_path, transform=model.full_transform)
    gen = torch.utils.data.DataLoader(
        image_folder,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False
    )
    print("Computing descriptors")
    img_list = image_folder.imgs
    n_steps = math.ceil(len(img_list) / batch_size)
    all_feats = model.predict_generator_with_netlvad(gen, n_steps=n_steps, verbose=verbose)
    if use_multi_resolution:
        for shape in input_shapes:
            print("Loading images at shape: {}".format(shape))
            image_folder = folder.ImageFolder(root=dataset_path, transform=model.get_transform(shape[0]))
            gen = torch.utils.data.DataLoader(
                image_folder,
                batch_size=batch_size,
                num_workers=8,
                shuffle=False,
            )
            print("Computing descriptors")
            all_feats += model.predict_generator_with_netlvad(gen, n_steps=n_steps, verbose=verbose)
    all_feats = normalize(all_feats)
    if pca is not None:
        if not os.path.isfile(pca):
            print("PCA {} not found. Starting training.".format(pca))
            train_pca(model, pca)

        print("Using PCA.")
        pca_dataset = h5py.File(pca, 'r')
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

    # nbrs = NearestNeighbors(n_neighbors=len(img_list), metric='cosine').fit(all_feats)
    # distances, indices = nbrs.kneighbors(all_feats)

    distances, indices = my_utils.torch_nn(all_feats, verbose=False)

    APs = []
    queries = {}
    # open queries
    if dataset == 'o':
        gt_path = "gt-oxford"
    else:
        gt_path = "gt-paris"
    text_files = os.listdir(gt_path)
    for file in text_files:
        if file.endswith("_query.txt"):
            query_file = open(gt_path + "/" + file, 'r')

            if dataset == 'o':
                query_pic = query_file.readline().split(" ")[0][len("oxc1_"):]
            else:
                query_pic = query_file.readline().split(" ")[0]

            query_name = file[:-len("_query.txt")]
            queries[query_pic] = query_name
    for row in indices:
        file_name, _ = img_list[row[0]]
        file_name = file_name.split("/")[-1].split(".")[0]
        if file_name in set(queries.keys()):
            ranked_list = []
            for j in row:
                file_name_j, _ = img_list[j]
                file_name_j = file_name_j.split("/")[-1].split(".")[0]
                ranked_list += [file_name_j]
            ap = compute_ap(query_name=queries[file_name], ranked_list=ranked_list, gt_path=gt_path)
            APs.append(ap)
    return APs


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

        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
        old_recall = recall
        old_precision = precision
        j += 1

    return ap


if __name__ == "__main__":
    main()
