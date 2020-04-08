import gc
import math
import os
import queue
import random
import threading
import time
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils import data

import netvlad_model as nm
import paths

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def show_triplet(triplet):
    fig = plt.figure(figsize=(10, 10))
    for i, t in enumerate(triplet):
        fig.add_subplot(1, 3, i + 1)
        t = t.astype('int')
        plt.imshow(t)

    plt.show()
    # plt.savefig("triplets/triplet{}.jpg".format(random.randint(1, 100000)))
    print("Triplet saved")
    time.sleep(5)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocess_input(x):
    # RGB -> BGR
    x = x[:, :, [2, 1, 0]]
    mean = [103.939, 116.779, 123.68]

    x[:, :, 0] -= mean[0]
    x[:, :, 1] -= mean[1]
    x[:, :, 2] -= mean[2]

    return x


def restore(x):
    mean = [103.939, 116.779, 123.68]
    y = np.array(x)
    y[..., 0] += mean[0]
    y[..., 1] += mean[1]
    y[..., 2] += mean[2]

    # 'BGR'->'RGB'
    y = y[:, :, [2, 1, 0]]
    return y


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def load_img(path, transform):
    img = Image.open(path)
    img = transform(img)
    # img.resize(resize_shape, Image.BILINEAR)
    return img


# def open_img(path, input_shape=nm.NetVladBase.input_shape):
#     img = image.load_img(path, target_size=(input_shape[0], input_shape[1]), interpolation='bilinear')
#     img = image.img_to_array(img)
#     img = preprocess_input(img)
#     img_id = path.split('/')[-1]
#
#     return img, img_id

class TripletDataset(data.DataLoader):
    def __init__(self, triplet_list, transform):
        super(TripletDataset, self).__init__()
        self.transform = transform
        self.triplet_list = triplet_list
        print("Number of triplets generated: ", self.__len__())

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, index):
        im_a = load_img(path=self.triplet_list[index][0], transform=self.transform)
        im_p = load_img(path=self.triplet_list[index][1], transform=self.transform)
        im_n = load_img(path=self.triplet_list[index][2], transform=self.transform)
        return im_a, im_p, im_n


class ImagesFromListDataset(data.Dataset):
    def __init__(self, transform, image_list, label_list=None):
        super(ImagesFromListDataset, self).__init__()
        self.transform = transform

        self.set_image_list(image_list, label_list)

    def set_image_list(self, image_list, label_list):
        if label_list is not None:
            assert len(image_list) == len(label_list)
            self.label_list = label_list

        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im = load_img(path=self.image_list[index], transform=self.transform)
        if hasattr(self, 'label_list'):
            return im, self.label_list[index]
        else:
            return im



def torch_nn(feats, verbose=True):
    feats = torch.Tensor(feats).cuda()
    if verbose:
        print("Mining - Computing distances")
    distances = (feats.mm(feats.t()))
    del feats
    if verbose:
        print("Mining - Computing indices")
    distances, indices = distances.sort(descending=True)
    return distances, indices


class LandmarkTripletGenerator():
    def __init__(self, train_dir, model, mining_batch_size=2048, minibatch_size=24, images_per_class=10,
                 semi_hard_prob=0.5, threshold=20, verbose=False, use_crop=False, print_statistics=False):


        esclude_classes_with_few_images = True

        self.print_statistics = print_statistics

        classes = os.listdir(train_dir)

        print("Available classes: {}".format(len(classes)))

        to_remove = []

        if esclude_classes_with_few_images:
            escluded_count = 0
            for class_ in classes:
                dir_path = os.path.join(train_dir, class_)
                n_images = len(os.listdir(dir_path))
                if n_images < images_per_class:
                    to_remove.append(class_)
                    escluded_count += n_images
                    # print("Class {} escluded from list with {} images".format(class_, n_images))

            for class_ in to_remove:
                classes.remove(class_)

            print("Available classes after esclusion: {}; Escluded images: {}".format(len(classes), escluded_count))

        self.classes = classes
        self.train_dir = train_dir
        self.mining_batch_size = mining_batch_size
        self.images_per_class = images_per_class
        self.use_crop = use_crop

        # self.loader = Loader(batch_size=mining_batch_size, classes=classes, n_classes=n_classes, train_dir=train_dir,
        #                      transform=model.full_transform)

        self.transform = model.full_transform
        self.minibatch_size = minibatch_size
        self.model = model
        self.verbose = verbose

        self.threshold = threshold
        self.semi_hard_prob = semi_hard_prob

        target_loss = 0.1
        delta = 0.04

        self.mining_iterations = 0

        self.loss_min = np.max(target_loss - delta, 0)
        self.loss_max = target_loss + delta

    def load_images_list(self, batch_size, classes, n_classes, train_dir):
        shuffled_classes = list(classes)
        random.shuffle(shuffled_classes)
        picked_classes = shuffled_classes[:n_classes]
        # print("Different classes in batch: ", n_classes)
        # load each image in those classes
        imgs = []
        for i, c in enumerate(picked_classes):
            images_in_c = os.listdir(train_dir + "/" + c)
            num_samples_in_c = len(images_in_c)
            if num_samples_in_c < batch_size // n_classes:
                print("LESS SAMPLES IN CLASS THAN TARGET ", num_samples_in_c)
            random.shuffle(images_in_c)
            images_in_c = images_in_c[:min(batch_size // n_classes, num_samples_in_c)]
            for image_in_c in images_in_c:
                class_index = classes.index(c)
                imgs += [(image_in_c, class_index, c)]
        # randomize the image list
        random.shuffle(imgs)
        # pick the first batch_size (if enough)
        batch_size_ = min(batch_size, len(imgs))
        imgs = imgs[:batch_size_]
        images_list = []
        label_list = []
        # load the images
        # print("Opening the images (producer thread)")
        for im, label, dir in imgs:
            # image = load_img(path=train_dir + "/" + dir + "/" + im, transform=self.transform)
            # image = image.unsqueeze(0)
            images_list.append(train_dir + "/" + dir + "/" + im)
            label_list.append(label)

        return images_list, label_list

    def generator(self):
        while True:

            current_mining_batch_size = self.mining_batch_size
            current_images_per_class = self.images_per_class

            self.mining_iterations += 1
            print("\nMining - iteration {}; Mining batch size: {}; Images per class: {}".format(self.mining_iterations,
                                                                                               current_mining_batch_size,
                                                                                               current_images_per_class))

            image_list, label_list = self.load_images_list(batch_size=current_mining_batch_size, classes=self.classes,
                                                           n_classes=current_mining_batch_size // current_images_per_class,
                                                           train_dir=self.train_dir)

            yield image_list, label_list


def evaluation_triplet_generator(train_dir, netbatch_size=32, model=None):
    classes = os.listdir(train_dir)

    # n_classes = batch_size // 4
    picked_classes = classes

    # load each image in those classes
    imgs = []
    for i, c in enumerate(picked_classes):
        images_in_c = os.listdir(train_dir + "/" + c)
        # images_in_c = zip(images_in_c, [i]*len(images_in_c), [c]*len(images_in_c))
        for image_in_c in images_in_c:
            imgs += [(image_in_c, i, c)]

    # randomize the image list
    random.shuffle(imgs)

    # pick the first batch_size (if enough)

    images_array = []
    label_array = []

    # load the images
    print("Opening the images (evaluation)")
    for im, index, dir in imgs:
        image, _ = load_img(path=train_dir + "/" + dir + "/" + im, transform=model.full_transform)
        label = index
        images_array.append(image)
        label_array.append(label)

    images_array = np.array(images_array)
    label_array = np.array(label_array)

    while True:
        print("New iteration (evaluation)")
        # pick n_classes from the dirs
        # random.shuffle(classes)

        print("Computing descriptors (evaluation)")
        feats = model.predict(images_array)
        feats = normalize(feats)

        nbrs = NearestNeighbors(n_neighbors=len(images_array), metric='l2').fit(feats)
        distances, indices = nbrs.kneighbors(feats)

        triplets = []

        # find triplets:
        print("Finding triplets (evaluation)")
        for i, row in enumerate(indices):
            anchor_label = label_array[i]

            j_neg = -1
            j_pos = -1

            for j, col in enumerate(row):
                # find first negative
                r_label = label_array[col]
                if (j_neg == -1) and (
                        r_label == anchor_label):  # scorre finchè non trova il primo negativo
                    j_pos = j
                    # continue
                elif (j_neg == -1) and (r_label != anchor_label):
                    j_neg = j
                elif (j_neg != -1) and (r_label == anchor_label):  # se ha trovato prima un negativo di un positivo
                    j_pos = j

            if (j_pos is not -1) and (j_neg is not -1):
                triplet = row[0], row[j_pos], row[j_neg]
                triplets.append(triplet)

            if False:
                print("Distance between indices", j_pos - j_neg)
                print("L2 distance between query and positive: ", distances[i][j_pos])
                print("L2 distance between query and negative: ", distances[i][j_neg])
                print("Triplete Loss (a=0.1): ", 0.1 + distances[i][j_pos] ** 2. - distances[i][j_neg] ** 2.)
                # for t in triplet:
                #    plt.imshow(images_array[t]*0.5 + 0.5)
                #    plt.show()
                i, j, k = triplet
                t_ = (images_array[i], images_array[j], images_array[k])
                show_triplet(t_)
                # time.sleep(6)

        im_triplets = [[images_array[i], images_array[j], images_array[k]] for i, j, k in triplets]
        random.shuffle(im_triplets)
        DIM = min(len(images_array), 256)

        # del images_array, indices, distances, feats
        # gc.collect()

        # print(len(im_triplets))
        im_triplets = im_triplets[:DIM]

        pages = math.ceil(DIM / netbatch_size)

        keep_epoch = 1
        for e in range(keep_epoch):
            for page in range(pages):
                triplets_out = im_triplets[page * netbatch_size: min((page + 1) * netbatch_size, DIM)]

                anchors = torch.cat([t[0].unsqueeze(0) for t in triplets_out], dim=0)
                positives = torch.cat([t[1].unsqueeze(0) for t in triplets_out], dim=0)
                negatives = torch.cat([t[2].unsqueeze(0) for t in triplets_out], dim=0)

                yield [anchors, positives, negatives], None  # , [y_fake]*3

        # y_fake = np.zeros((len(images_array), net_output + 1))
        # yield [images_array, label_array], y_fake


import yaml


def main():
    conf_file = open('best-model-conf.yaml', 'r')
    conf = dict(yaml.safe_load(conf_file))
    conf_file.close()

    use_power_norm = conf['use_power_norm']
    use_multi_resolution = conf['use_multi_resolution']
    side_res = conf['input-shape']

    network_conf = conf['network']
    net_name = network_conf['name']

    nm.NetVladBase.input_shape = (side_res, side_res, 3)
    if use_multi_resolution:
        nm.NetVladBase.input_shape = (None, None, 3)

    my_model = None
    if net_name == "vgg":
        my_model = nm.NetVLADSiameseModel(**network_conf)
    elif net_name == "resnet":
        my_model = nm.NetVladResnet(**network_conf)
    else:
        print("Network name not valid.")

    vgg_netvlad = my_model.build_netvladmodel()
    weight_name = "best_model.h5"
    print("Loading weights: " + weight_name)
    vgg_netvlad.load_weights(weight_name)
    vgg_netvlad = my_model.get_netvlad_extractor()

    # vgg_netvlad = ResNet50(input_shape=(336,336,3), weights='imagenet', pooling='avg', include_top=False)

    landmark_generator = LandmarkTripletGenerator(train_dir=paths.landmark_clustered_path,
                                                  model=vgg_netvlad,
                                                  mining_batch_size=2048,
                                                  minibatch_size=6, semi_hard_prob=0.0,
                                                  threshold=5, verbose=True)

    train_generator = landmark_generator.generator()

    for el in train_generator:
        x, y = el
        a, p, n = x
        print("Number of triplets: ", a.shape[0])
        for i in range(a.shape[0]):
            print("Triplet loss: ", y[i])

            descs = vgg_netvlad.predict(np.array([a[i], p[i], n[i]]))
            descs = normalize(descs)
            a_desc = descs[0]
            p_desc = descs[1]
            n_desc = descs[2]

            alpha = 0.1
            d_a_p = np.linalg.norm(a_desc - p_desc, ord=2)
            d_a_n = np.linalg.norm(a_desc - n_desc, ord=2)

            loss = alpha + d_a_p - d_a_n

            print("Loss: ", loss)
            show_triplet([restore(a[i]), restore(p[i]), restore(n[i])])


if __name__ == '__main__':
    main()

# model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
# generator = LandmarkTripletGenerator("/mnt/m2/dataset/", model=model)
# custom_generator = generator.generator()
#
# show_triplets = True
# for el in custom_generator:
#     if show_triplets:
#         x = el[0]
#         a, p, n = x
#         print("Number of triplets: ", a.shape[0])
#         for i in range(a.shape[0]):
#             show_triplet([a[i], p[i], n[i]])
