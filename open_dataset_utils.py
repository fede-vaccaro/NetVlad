import os

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from netvlad_model import input_shape


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.txt')]


def open_img(path, input_shape=input_shape):
    img = image.load_img(path, target_size=(input_shape[0], input_shape[1]))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img_id = path.split('/')[-1]

    return img, img_id


def image_generator(files, index, classes, net_output=0, batch_size=64, input_shape=input_shape, augmentation=False):
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


def custom_generator_from_keras(train_dir, batch_size=32, net_output=0, train_classes=None):
    if train_classes is None:
        image_generator = ImageDataGenerator(rescale=1. / 255.)
        """
                                             , rotation_range=45,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=False,
                                             fill_mode='nearest')"""
    else:
        image_generator = ImageDataGenerator(rescale=1. / 255.)

    data_generator = image_generator.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(input_shape[0], input_shape[1]),
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
            y_fake = np.zeros((len(x_), net_output + 1))

            # if train_classes is not None:
            #    classes_diff = train_classes - data_generator.num_classes
            #    y_diff = np.zeros((len(y_), classes_diff))
            #    y_ = np.hstack((y_, y_diff))
            #    y_fake = np.hstack((y_fake, y_diff))

            if net_output is not None:
                yield ([x_, y_], y_fake)
            else:
                yield x_

    return generator(), data_generator.samples, data_generator.num_classes

import random

def landmark_generator(train_dir, batch_size=256, net_output=0):
    classes = os.listdir(train_dir)

    n_classes = batch_size // 4

    while True:
        # pick n_classes from the dirs
        random.shuffle(classes)
        picked_classes = classes[:n_classes]

        # load each image in those classes
        imgs = []
        for i, c in enumerate(picked_classes):
            images_in_c = os.listdir(train_dir + "/" + c)
            #images_in_c = zip(images_in_c, [i]*len(images_in_c), [c]*len(images_in_c))
            for image_in_c in images_in_c:
                imgs += [(image_in_c, i, c)]

        # randomize the image list
        random.shuffle(imgs)

        # pick the first 256 (if enough)
        batch_size_ = min(batch_size, len(imgs))
        imgs = imgs[:batch_size_]

        images_array = []
        label_array = []

        # load the images
        for im, index, dir in imgs:
            image, _ = open_img(train_dir + "/" + dir + "/" + im, input_shape=input_shape)
            label = index
            images_array.append(image)
            label_array.append(label)

        images_array = np.array(images_array)
        label_array = np.array(label_array)

        y_fake = np.zeros((len(images_array), net_output + 1))

        yield [images_array, label_array], y_fake

# index, classes = generate_index_ukbench('ukbench')
# print(len(index), len(classes))

# for k in sorted(index.keys()):
#    print(k, index[k])
# custom_generator,_ ,_  = custom_generator_from_keras("partition", 64, net_output = int(32e3))

custom_generator = landmark_generator("partition_0", net_output=32000)

#for el in custom_generator:
#   print(el[0][0].shape, el[0][1].shape, el[1].shape)
