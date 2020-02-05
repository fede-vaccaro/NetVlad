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

import netvlad_model as nm
import paths


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.txt')]


def open_img(path, input_shape=nm.NetVladBase.input_shape):
    img = image.load_img(path, target_size=(input_shape[0], input_shape[1]), interpolation='bilinear')
    # img = (image.img_to_array(img) - 127.5) / 127.5
    img = preprocess_input(image.img_to_array(img))
    img_id = path.split('/')[-1]

    return img, img_id


def image_generator(files, index, classes, net_output=0, batch_size=64, input_shape=nm.NetVladBase.input_shape,
                    augmentation=False):
    train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range=60,
                                       width_shift_range=0.4,
                                       height_shift_range=0.4,
                                       shear_range=0.1,
                                       zoom_range=0.4,
                                       horizontal_flip=False,
                                       fill_mode='nearest')

    while True:
        batch_paths = np.random.choice(a=files,
                                       size=batch_size)

        x_batch = []
        label_batch = []

        for input_path in batch_paths:
            img, id = open_img(input_path, input_shape=input_shape)
            x_batch += [img]

            tags = np.zeros(len(classes))
            for i, c in enumerate(classes):
                if c in index[id]:
                    tags[i] = 1
            label_batch += [tags]

        y_batch = np.zeros((batch_size, net_output + len(classes)))
        x_batch = np.array(x_batch)
        label_batch = np.array(label_batch)

        # label_cross = np.dot(label_batch, label_batch.T)
        # label_cross_bool = label_cross.astype('bool')
        if net_output is not 0:
            if augmentation:
                generator_augmentation = train_datagen.flow(x_batch, label_batch, batch_size=batch_size, shuffle=True)
                x_batch, label_batch = next(generator_augmentation)
                yield ([x_batch, label_batch], y_batch)
        else:
            yield x_batch


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


def generate_index_ukbench(path):
    imnames = get_imlist(path)
    image_dict = {}
    classes = set()

    to_strip_front = len(".jpg")
    to_strip_back = len("ukbench/ukbench")

    for name in imnames:
        id = name[:-to_strip_front]
        id = id[to_strip_back:]
        id_int = int(id)
        query = (id_int) // 4

        image_dict[name[len("ukbench/"):]] = [str(query)]
        classes.add(str(query))

    return image_dict, list(classes)


def generate_index_holidays(path):
    # relevant_tags_txt = get_txtlist(path)

    images_dict = {}
    classes = set()

    labeled_file = open(path, "r")

    for line in labeled_file.readlines():
        split = line.split(" ")[:3]

        img_name = split[0]
        img_query = split[2]

        if images_dict.keys().__contains__(img_name):
            images_dict[img_name].append(img_query)
        else:
            images_dict[img_name] = [img_query]

        classes.add(img_query)

    return images_dict, sorted(list(classes))


def custom_generator_from_keras(train_dir, batch_size=32, net_output=None, train_classes=None):
    if train_classes is None:
        image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
        """
                                             , rotation_range=45,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=False,
                                             fill_mode='nearest')"""
    else:
        image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    data_generator = image_generator.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(nm.NetVladBase.input_shape[0], nm.NetVladBase.input_shape[1]),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical', shuffle=False)
    """
    else:
        data_generator = image_generator.flow_from_directory(
            # This is the target directory
            train_dir,
            # All images will be resized to 150x150
            target_size=(input_shape[0], input_shape[1]),
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical', shuffle=True)
        generators.append(data_generator)
    """
    print("samples: ", data_generator.samples)

    def generator():
        i = 0
        while True:
            # i = i % len(generators)
            x_, y_ = next(data_generator)
            y_ = np.argmax(y_, axis=1)
            i = (i + 1) % len(data_generator)
            # print(i)

            # if train_classes is not None:
            #    classes_diff = train_classes - data_generator.num_classes
            #    y_diff = np.zeros((len(y_), classes_diff))
            #    y_ = np.hstack((y_, y_diff))
            #    y_fake = np.hstack((y_fake, y_diff))

            if net_output is not None:
                y_fake = np.zeros((len(x_), net_output + 1))
                yield ([x_, y_], y_fake)
            else:
                yield x_

    return generator(), data_generator.samples, data_generator.num_classes


class Loader(threading.Thread):
    def __init__(self, batch_size, classes, n_classes, train_dir):
        self.batch_size = batch_size
        self.classes = classes
        self.n_classes = n_classes
        self.train_dir = train_dir

        self.keep_loading = True

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
            # load the images
            # print("Opening the images (producer thread)")
            for im, index, dir in imgs:
                image, _ = open_img(train_dir + "/" + dir + "/" + im, input_shape=nm.NetVladBase.input_shape)
                label = index
                images_array.append(image)
                label_array.append(label)

            # print("Images loaded")
            if self.keep_loading:
                self.q.put((np.array(images_array), np.array(label_array)))

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
    def __init__(self, train_dir, mining_batch_size=2048, minibatch_size=24, model=None, use_multiprocessing=True,
                 semi_hard_prob=0.5, threshold=20, verbose=True):
        classes = os.listdir(train_dir)

        n_classes = mining_batch_size // 4

        self.loader = Loader(mining_batch_size, classes, n_classes, train_dir)
        if use_multiprocessing:
            self.loader.start()

        self.use_multiprocessing = use_multiprocessing
        self.minibatch_size = minibatch_size
        self.model = model
        self.verbose = False

        self.threshold = threshold
        self.semi_hard_prob = semi_hard_prob

        self.loss_min = 0.09
        self.loss_max = 0.11

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
            losses = []

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
                        if j > 1:
                            j_neg = j
                        else:
                            break
                        if j_neg > 1 and (np.random.uniform() > 1.0 - self.semi_hard_prob):
                            j_pos = j_neg - 1
                    elif (j_neg != -1) and (r_label == anchor_label):
                        j_pos = j

                    # if (j_pos is not -1) and (j_neg is not -1) and (j_pos - j_neg < self.threshold):
                    if (j_pos is not -1) and (j_neg is not -1):
                        triplet = row[0], row[j_pos], row[j_neg], r_label

                        d_a_p = distances[i][j_pos]
                        d_a_n = distances[i][j_neg]

                        loss = 0.1 + d_a_p ** 2 - d_a_n ** 2

                        if self.loss_min < loss < self.loss_max:
                            triplets.append(triplet)
                            losses.append(loss)

            select_triplets_per_class = False
            if select_triplets_per_class:
                class_set = []
                selected_triplets = []
                selected_losses = []

                for i, t in enumerate(triplets):
                    label = t[3]
                    if label in class_set:
                        continue
                    else:
                        selected_triplets += [t[:3]]
                        selected_losses += [losses[i]]
                        class_set += [label]

                triplets = selected_triplets
                losses = selected_losses
            if self.verbose:
                print("Different classes: {}".format(len(class_set)))

            im_triplets = [[images_array[i], images_array[j], images_array[k]] for i, j, k, _ in triplets]
            # random.shuffle(im_triplets)

            # del images_array, indices, distances, feats
            # gc.collect()

            # select just K different classes
            # K_classes = 256
            K_classes = len(im_triplets)
            # K_classes = min(K_classes, len(class_set))
            # im_triplets = im_triplets[:K_classes]

            pages = math.ceil(K_classes / self.minibatch_size)
            for page in range(pages):
                triplets_out = im_triplets[page * self.minibatch_size: min((page + 1) * self.minibatch_size, K_classes)]

                anchors = np.array([t[0] for t in triplets_out])
                positives = np.array([t[1] for t in triplets_out])
                negatives = np.array([t[2] for t in triplets_out])

                yield [anchors, positives, negatives], np.array(losses)  # , [y_fake]*3


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
        image, _ = open_img(train_dir + "/" + dir + "/" + im, input_shape=nm.NetVladBase.input_shape)
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
import yaml


def show_triplet(triplet):
    fig = plt.figure(figsize=(10, 10))
    for i, t in enumerate(triplet):
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(t * 0.5 + 0.5)

    plt.show()
    time.sleep(3)


def main():
    conf_file = open('resnet-conf.yaml', 'r')
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

    landmark_generator = LandmarkTripletGenerator(train_dir=paths.landmarks_path,
                                                  model=vgg_netvlad,
                                                  mining_batch_size=2048,
                                                  minibatch_size=6, semi_hard_prob=.0,
                                                  threshold=5)

    train_generator = landmark_generator.generator()

    for el in train_generator:
        x, y = el
        a, p, n = x
        print("Number of triplets: ", a.shape[0])
        for i in range(a.shape[0]):
            print("Triplet loss: ", y[i])
            if y[i] > 0.1:
                print("Hard triplet")
            else:
                print("Semi-hard triplet")
            show_triplet([a[i], p[i], n[i]])


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
