import gc
import math
import os
import queue
import random
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from netvlad_model import input_shape

import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.txt')]


def open_img(path, input_shape=input_shape):
    img = image.load_img(path, target_size=(input_shape[0], input_shape[1]))
    # img = (image.img_to_array(img) - 127.5) / 127.5
    img = preprocess_input(image.img_to_array(img))
    img_id = path.split('/')[-1]

    return img, img_id


def generate_index_mirflickr(path):
    relevant_tags_txt = get_txtlist(path)
    images_dict = {}
    classes = []
    for tag_txt_name in relevant_tags_txt:
        labeled_file = open(tag_txt_name, "r")

        tag_name = tag_txt_name[len(path) + 1:-4]

        classes.append(tag_name)
        for img_name in labeled_file.readlines():

            img_name = "im" + str(int(img_name)) + ".jpg"

            if images_dict.keys().__contains__(img_name):
                images_dict[img_name].append(tag_name)
            else:
                images_dict[img_name] = [tag_name]
    return images_dict, classes


class Loader(threading.Thread):
    def __init__(self, batch_size, classes, n_classes, train_dir, yield_paths=False):
        self.batch_size = batch_size
        self.classes = classes
        self.n_classes = n_classes
        self.train_dir = train_dir

        self.keep_loading = True
        self.yield_paths = yield_paths

        self.q = queue.Queue(4)
        super(Loader, self).__init__()

    def load_batch(self, batch_size, classes, n_classes, train_dir):
        if not self.q.full():
            random.shuffle(classes)
            picked_classes = classes[:n_classes]
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
            batch_size_ = min(batch_size, len(imgs))
            imgs = imgs[:batch_size_]

            images_array = []
            label_array = []
            path_array = []

            # load the images
            print("Opening the images (producer thread)")
            for im, index, dir in imgs:
                image, _ = open_img(train_dir + "/" + dir + "/" + im, input_shape=input_shape)
                label = index
                images_array.append(image)
                label_array.append(label)
                path_array.append(dir + "/" + im)

            # print("Images loaded")
            if self.keep_loading:
                self.q.put((np.array(images_array), np.array(label_array), path_array))

            # self.q.task_done()
            # print("Object enqueued: ", (self.q.qsize()))
            gc.collect()
            # return images_array, label_array

    # self.images_array = np.array(images_array)
    # self.label_array = np.array(label_array)

    def run(self) -> None:
        while self.keep_loading:
            self.load_batch_()

    def load_batch_(self):
        self.load_batch(self.batch_size, self.classes, self.n_classes, self.train_dir)

    def stop_loading(self):
        self.keep_loading = False
        self.q.queue.clear()


class LandmarkTripletGenerator():
    def __init__(self, train_dir, mining_batch_size=2048, minibatch_size=24, model=None, use_multiprocessing=True):
        classes = os.listdir(train_dir)

        n_classes = mining_batch_size // 4

        self.loader = Loader(mining_batch_size, classes, n_classes, train_dir)
        if use_multiprocessing:
            self.loader.start()

        self.use_multiprocessing = use_multiprocessing
        self.minibatch_size = minibatch_size
        self.model = model
        self.verbose = False

    def generator(self):
        while True:
            if self.verbose:
                print("New mining iteration")
            # pick n_classes from the dirs

            # images_array, label_array = load_batch(batch_size, classes, n_classes, train_dir)
            # if loader.q.empty():
            #   loader.q.join()
            if not self.use_multiprocessing:
                self.loader.load_batch_()
            images_array, label_array = self.loader.q.get()
            if self.verbose:
                print("Computing descriptors (mining)")
            feats = self.model.predict(images_array)
            feats = normalize(feats)

            nbrs = NearestNeighbors(n_neighbors=len(images_array), metric='l2').fit(feats)
            distances, indices = nbrs.kneighbors(feats)

            triplets = []

            # find triplets:
            if self.verbose:
                print("Finding triplets (mining)")
            for i, row in enumerate(indices):
                anchor_label = label_array[i]

                j_neg = -1
                j_pos = -1

                for j, col in enumerate(row):
                    # find first negative
                    r_label = label_array[col]
                    if (j_pos == -1) and (j_neg == -1) and (
                            r_label == anchor_label):  # scorre finchè non trova il primo negativo
                        continue
                    elif (j_neg == -1) and (r_label != anchor_label):
                        j_neg = j
                        if j_neg > 1 and (np.random.uniform() > 0.5):
                            j_pos = j_neg - 1
                    elif (j_neg != -1) and (r_label == anchor_label):
                        j_pos = j

                    if (j_pos is not -1) and (j_neg is not -1) and (j_pos - j_neg < 20):
                        triplet = row[0], row[j_pos], row[j_neg], r_label
                        triplets.append(triplet)

                        if False:
                            print("Distance between indices (p:{}, n:{}) : {}".format(j_pos, j_neg, j_pos - j_neg))
                            # print("L2 distance between query and positive: ", distances[i][j_pos])
                            # print("L2 distance between query and negative: ", distances[i][j_neg])
                            # print("Triplete Loss (a=0.1): ", 0.1 + distances[i][j_pos] ** 2. - distances[i][j_neg] ** 2.)
                            i, j, k = triplet
                            t_ = (images_array[i], images_array[j], images_array[k])
                            show_triplet(t_)
                        break

            # select triplets per classes
            class_set = []
            selected_triplets = []

            for t in triplets:
                label = t[3]
                if label in class_set:
                    continue
                else:
                    selected_triplets += [t[:3]]
                    class_set += [label]

            triplets = selected_triplets
            if self.verbose:
                print("Different classes: {}".format(len(class_set)))

            im_triplets = [[images_array[i], images_array[j], images_array[k]] for i, j, k in triplets]
            random.shuffle(im_triplets)

            # del images_array, indices, distances, feats
            # gc.collect()

            # select just K different classes
            K_classes = 256
            K_classes = min(K_classes, len(class_set))
            im_triplets = im_triplets[:K_classes]

            pages = math.ceil(K_classes / self.minibatch_size)
            for page in range(pages):
                triplets_out = im_triplets[page * self.minibatch_size: min((page + 1) * self.minibatch_size, K_classes)]

                anchors = np.array([t[0] for t in triplets_out])
                positives = np.array([t[1] for t in triplets_out])
                negatives = np.array([t[2] for t in triplets_out])

                yield [anchors, positives, negatives], None  # , [y_fake]*3


class LandmarkMiner():
    def __init__(self, train_dir, model, mining_batch_size=2048, use_multiprocessing=True):
        classes = os.listdir(train_dir)

        n_classes = mining_batch_size // 4

        self.loader = Loader(mining_batch_size, classes, n_classes, train_dir, yield_paths=True)
        if use_multiprocessing:
            self.loader.start()

        self.use_multiprocessing = use_multiprocessing
        self.model = model
        self.verbose = True

    def generator(self):
        while True:
            if self.verbose:
                print("New mining iteration")
            # pick n_classes from the dirs

            # images_array, label_array = load_batch(batch_size, classes, n_classes, train_dir)
            # if loader.q.empty():
            #   loader.q.join()
            if not self.use_multiprocessing:
                self.loader.load_batch_()
            images_array, label_array, path_array = self.loader.q.get()
            if self.verbose:
                print("Computing descriptors (mining)")
            feats = self.model.predict(images_array)
            feats = normalize(feats)

            nbrs = NearestNeighbors(n_neighbors=len(images_array), metric='l2').fit(feats)
            distances, indices = nbrs.kneighbors(feats)

            triplets = []

            # find triplets:
            if self.verbose:
                print("Finding triplets (mining)")
            for i, row in enumerate(indices):
                anchor_label = label_array[i]

                j_neg = -1
                j_pos = -1

                for j, col in enumerate(row):
                    # find first negative
                    r_label = label_array[col]
                    if (j_pos == -1) and (j_neg == -1) and (
                            r_label == anchor_label):  # scorre finchè non trova il primo negativo
                        continue
                    elif (j_neg == -1) and (r_label != anchor_label):
                        j_neg = j
                        if j_neg > 1 and (np.random.uniform() > 0.5):
                            j_pos = j_neg - 1
                    elif (j_neg != -1) and (r_label == anchor_label):
                        j_pos = j

                    if (j_pos is not -1) and (j_neg is not -1) and (j_pos - j_neg < 20):
                        triplet = row[0], row[j_pos], row[j_neg], r_label
                        triplets.append(triplet)

                        if False:
                            print("Distance between indices (p:{}, n:{}) : {}".format(j_pos, j_neg, j_pos - j_neg))
                            # print("L2 distance between query and positive: ", distances[i][j_pos])
                            # print("L2 distance between query and negative: ", distances[i][j_neg])
                            # print("Triplete Loss (a=0.1): ", 0.1 + distances[i][j_pos] ** 2. - distances[i][j_neg] ** 2.)
                            i, j, k = triplet
                            t_ = (images_array[i], images_array[j], images_array[k])
                            show_triplet(t_)
                        break

            # select triplets per classes
            class_set = []
            selected_triplets = []

            for t in triplets:
                label = t[3]
                if label in class_set:
                    continue
                else:
                    selected_triplets += [t[:3]]
                    class_set += [label]

            triplets = selected_triplets
            if self.verbose:
                print("Different classes: {}".format(len(class_set)))

            im_triplets = [[path_array[i], path_array[j], path_array[k]] for i, j, k in triplets]
            random.shuffle(im_triplets)

            yield im_triplets


def holidays_triplet_generator(train_dir, netbatch_size=32, model=None):
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
        image, _ = open_img(train_dir + "/" + dir + "/" + im, input_shape=input_shape)
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
        DIM = len(images_array)

        # del images_array, indices, distances, feats
        # gc.collect()

        # print(len(im_triplets))
        im_triplets = im_triplets[:DIM]

        pages = math.ceil(DIM / netbatch_size)

        keep_epoch = 1
        for e in range(keep_epoch):
            for page in range(pages):
                triplets_out = im_triplets[page * netbatch_size: min((page + 1) * netbatch_size, DIM)]

                anchors = np.array([t[0] for t in triplets_out])
                positives = np.array([t[1] for t in triplets_out])
                negatives = np.array([t[2] for t in triplets_out])

                yield [anchors, positives, negatives], None  # , [y_fake]*3

        # y_fake = np.zeros((len(images_array), net_output + 1))
        # yield [images_array, label_array], y_fake


# index, classes = generate_index_ukbench('ukbench')
# print(len(index), len(classes))

# for k in sorted(index.keys()):
#    print(k, index[k])
# custom_generator,_ ,_  = custom_generator_from_keras("partition", 64, net_output = int(32e3))

# custom_generator = holidays_triplet_generator("holidays_small_", model=model)


def show_triplet(triplet):
    fig = plt.figure(figsize=(10, 10))
    for i, t in enumerate(triplet):
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(t * 0.5 + 0.5)

    plt.show()
    time.sleep(6)


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

from netvlad_model import NetVLADSiameseModel
import pandas as pd


def triplets_from_csv(csv_path, train_dir, batch_size):
    with open(csv_path, 'r') as csv:
        csv = pd.read_csv(csv_path, index_col=[0])
        while True:
            samples = csv.sample(batch_size)
            anchors, positives, negatives = [], [], []
            for id, row in samples.iterrows():
                a = id
                p = row[0]
                n = row[1]

                print(a, p, n)

                anchors.append(open_img(train_dir + a))
                positives.append(open_img(train_dir + p))
                negatives.append(open_img(train_dir + n))
            yield np.array(anchors), np.array(positives), np.array(negatives)


import paths


def main():
    my_model = NetVLADSiameseModel()
    vgg_netvlad = my_model.build_netvladmodel()
    model_name = "model_e94_sc-adam_0.0709_wu.h5"
    vgg_netvlad.load_weights(model_name)
    vgg_netvlad = my_model.get_netvlad_extractor()
    miner = LandmarkMiner(paths.landmarks_path, model=vgg_netvlad, mining_batch_size=2048*3)
    generator = miner.generator()

    n_images = 63000 * 4
    # n_images = 2048
    triplets = {}

    while len(triplets.keys()) < n_images:
        triplets_ = next(generator)
        for a, p, n in triplets_:
            triplets[a] = (p, n)
        print("Triplets count: {}/{}".format(len(triplets.keys()), n_images))

    filename = "landmark_triplets.csv"
    print("Triplets generated. Saving to ", filename)
    with open(filename, 'w') as file:
        for a in triplets.keys():
            p, n = triplets[a]
            file.write("{},{},{}\n".format(a, p, n))

    miner.loader.stop_loading()


if __name__ == '__main__':
    main()

