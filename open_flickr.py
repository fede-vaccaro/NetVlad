import os
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import time
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]

def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.txt')]


def open_img(path):
    input_shape = (224,224,3)
    img = image.load_img(path, target_size=(input_shape[0], input_shape[1]))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img_id = path.split('/')[-1]

    return img, img_id


def image_generator(files, index, classes, net_output, batch_size = 64):
    while True:
        batch_paths = np.random.choice(a=files,
                                       size=batch_size)

        x_batch = []
        label_batch = []

        for input_path in batch_paths:
            img, id = open_img(input_path)
            x_batch += [img]

            tags = np.zeros(len(classes))
            for i, c in enumerate(classes):
                if c in index[id]:
                    tags[i] = 1
            label_batch += [tags]

        y_batch = np.zeros((batch_size, net_output + len(classes)))
        x_batch = np.array(x_batch)
        label_batch = np.array(label_batch)

        label_cross = np.dot(label_batch, label_batch.T)
        label_cross_bool = label_cross.astype('bool')

        yield([x_batch, label_batch], y_batch)


def generate_index(path):
    relevant_tags_txt = get_txtlist(path)
    images_dict = {}
    values = []
    for tag_txt_name in relevant_tags_txt:
        labeled_file = open(tag_txt_name, "r")

        tag_name = tag_txt_name[len(path)+1:-4]

        values.append(tag_name)
        for img_name in labeled_file.readlines():

            img_name = "im" + str(int(img_name)) + ".jpg"

            if images_dict.keys().__contains__(img_name):
                images_dict[img_name].append(tag_name)
            else:
                images_dict[img_name] = [tag_name]

    return images_dict, values

dict, classes = generate_index('mirflickr_annotations')
print(len(dict.keys()))
print(classes)

files = list(dict.keys())
files = ["mirflickr/" + k for k in files]

gen = image_generator(files=files, index=dict, classes=classes, net_output=4000, batch_size=16)

for el in gen:
    [x_batch, label_batch], y_batch = el
    #print(x_batch.shape, label_batch.shape, y_batch.shape)
    #time.sleep(1)

