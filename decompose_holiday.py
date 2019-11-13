import random
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import math

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

triplets = []

for key in sorted(all_images.keys()):
    line_index, is_query = all_images[key]
    q = key

    line = lines[line_index]
    if is_query:
        p = line[len(line) - 1]
    else:
        p = line[0]

    temp_all_images_keys = list(all_images.keys())
    temp_all_images_keys.remove(key)
    temp_all_images_keys.remove(p)
    n = random.choice(temp_all_images_keys)

    triplet = (q, p, n)
    triplets.append(triplet)

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


show_triplets = False
if show_triplets:
    for triplet in triplets:
        imfiles = list("holidays_small/" + x for x in triplet)
        montage_im = montage(imfiles, thumb_size=(100, 100), ok=[True, True, False], shape=(1, 3))
        plt.imshow(montage_im)
        plt.show()
