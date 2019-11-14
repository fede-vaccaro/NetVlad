import math
import random

import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

path = "eval_holidays/perfect_result.dat"

f = open(path, 'r')

lines = []

# read lines from f
for line in f.readlines():
    lines.append(line.split(" "))

f.close()

# deletes indices from each line (they are in odd positions)

lines_no_indices = []

for line in lines:
    new_line = [el for i, el in enumerate(line) if i % 2 == 0]

    # remove '\n' from last element
    new_line[len(new_line) - 1] = new_line[len(new_line) - 1].strip('\n')
    lines_no_indices.append(new_line)
lines = lines_no_indices

all_images = {}

for i, line in enumerate(lines):
    line_index = i
    for j, img in enumerate(line):
        is_query = (j is 0)
        all_images[img] = (i, is_query)

# open images
input_shape = (224, 224, 3)


def images_to_tensor(imnames):
    images_array = []
    # open all images
    for name in imnames:
        img_path = 'holidays_small/' + name
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        images_array.append(img)
    images_array = np.array(images_array)
    images_array = vgg16.preprocess_input(images_array)
    print(images_array.shape)
    # images_array = preprocess_input(images_array)
    return images_array


keys = all_images.keys()
im_tensor = images_to_tensor(keys)
print("extracting descriptors")
vgg = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
descriptors = vgg.predict(im_tensor)

all_descriptors = {}
for i, key in enumerate(keys):
    all_descriptors[key] = descriptors[i]

print("fitting k nn")
nbrs = NearestNeighbors(n_neighbors=1491, metric='cosine').fit(descriptors)

triplets = []
unsupervised = False
if not unsupervised:
    for key in sorted(all_images.keys()):
        line_index, is_query = all_images[key]
        q = key

        line = lines[line_index]
        if is_query:
            p = line[len(line) - 1]
        else:
            permutation = [ind for ind in range(len(line))]
            permutation.remove(line.index(q))
            random.shuffle(permutation)
            p = line[permutation[0]]
        # print("P choiced for ", key)

        # temp_all_images_keys = list(all_images.keys())
        # temp_all_images_keys.remove(key)
        # temp_all_images_keys.remove(p)

        distances, indices = nbrs.kneighbors(np.expand_dims(all_descriptors[q], axis=0))
        indices = indices[0, :]

        n = "empty"

        for ind in indices[10:]:
            key_ = list(all_descriptors.keys())[ind]
            if key_ not in line:
                n = key_
                break

        # print("N choiced for ", key)
        # n = random.choice(temp_all_images_keys)

        triplet = (q, p, n)
        triplets.append(triplet)
else:
    for key in sorted(all_images.keys()):
        line_index, is_query = all_images[key]

        q = key

        distances, indices = nbrs.kneighbors(np.expand_dims(all_descriptors[q], axis=0))
        indices = indices[0, :]

        p = list(all_descriptors.keys())[indices[2]]
        n = list(all_descriptors.keys())[indices[5]]

        triplet = (q, p, n)
        triplets.append(triplet)

random.shuffle(triplets)
# output triplets
triplets_file = open("triplets.dat", 'w')
for triplet in triplets:
    str = ""
    for t in triplet:
        str += t + " "
    triplets_file.write(str + "\n")
triplets_file.close()


def montage(imfiles, thumb_size=(100, 100), ok=None, shape=None):
    # this function will create an image with thumbnailed version of imfiles.
    # optionally the user can provide an ok list such that len(ok)==len(imfiles) to differentiate correct from wrong results
    # optionally the user can provide a shape function which shapes the montage otherwise a square image is created.
    images = [PIL.Image.open(imname).resize(thumb_size, PIL.Image.BILINEAR) for imname in imfiles]
    # create a big image to contain all images
    if shape is None:
        n = int(math.sqrt(len(imfiles)))
        m = n
    else:
        n = shape[0]
        m = shape[1]
    new_im = PIL.Image.new('RGB', (m * thumb_size[0], n * thumb_size[0]))
    k = 0
    for i in range(0, n * thumb_size[0], thumb_size[0]):
        for j in range(0, m * thumb_size[0], thumb_size[0]):
            region = (j, i)
            if ok is not None:
                if ok[k]:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                if k > 0:
                    imar = np.array(images[k], dtype=np.uint8)
                    imar[0:5, :, :] = color
                    imar[:, 0:5, :] = color
                    imar[-5:, :, :] = color
                    imar[:, -5:, :] = color
                    images[k] = PIL.Image.fromarray(imar)
            new_im.paste(images[k], box=region)
            k += 1
    return new_im


show_triplets = True
if show_triplets:
    for triplet in triplets:
        imfiles = list("holidays_small/" + x for x in triplet)
        montage_im = montage(imfiles, thumb_size=(100, 100), ok=[True, True, False], shape=(1, 3))
        plt.imshow(montage_im)
        plt.show()
