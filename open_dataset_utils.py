import os

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from netvlad_model import input_shape
from keras.preprocessing.image import ImageDataGenerator

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
        query = (id_int)//4

        image_dict[name[len("ukbench/"):]] = [str(query)]
        classes.add(str(query))

    return image_dict, list(classes)



def generate_index_holidays(path):
    #relevant_tags_txt = get_txtlist(path)

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


def custom_generator_from_keras(train_dir, batch_size=32, net_output=0, n_classes=None, train_set=True):
    if train_set:
        train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range=60,
                                       width_shift_range=0.4,
                                       height_shift_range=0.4,
                                       shear_range=0.4,
                                       zoom_range=0.4,
                                       horizontal_flip=False,
                                       fill_mode='nearest')
    else:
        train_datagen = ImageDataGenerator(rescale=1./255.)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

    print("samples: ", train_generator.samples)


    while True:
        x_, y_ = next(train_generator)
        y_fake = np.zeros((len(x_), net_output + n_classes))

        if net_output is not None:
            yield ([x_, y_], y_fake)
        else:
            yield x_

#index, classes = generate_index_ukbench('ukbench')
#print(len(index), len(classes))

#for k in sorted(index.keys()):
#    print(k, index[k])
#custom_generator = custom_generator_from_keras("seefood/train", 32, net_output = int(32e3), n_classes=2)

#for el in custom_generator:
#    print(el[0][0].shape, el[0][1].shape, el[1].shape)
