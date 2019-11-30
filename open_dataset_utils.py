import os

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
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


def image_generator(files, index, classes, net_output=0, batch_size=64, input_shape=input_shape):

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

#index, classes = generate_index_ukbench('ukbench')
#print(len(index), len(classes))

#for k in sorted(index.keys()):
#    print(k, index[k])
