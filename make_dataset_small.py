import os
from PIL import Image


def get_im_list(path):
    return [(os.path.join(path, impath), impath) for impath in os.listdir(path)]

path = '/mnt/sdb-seagate/datasets/paris/'
path_dest = '/mnt/sdb-seagate/datasets/paris_small/'
im_list = get_im_list(path)

i = 0
for path, id in im_list:
    try:
        img = Image.open(path)
        width, height = img.size

        a = 640
        b = 480
        if width > height:
            img = img.resize((a, b), Image.ANTIALIAS)
        else:
            img = img.resize((b, a), Image.ANTIALIAS)
        img.save(path_dest + id, 'JPEG', quality=95)

        if i % 100 == 0:
            print("Image {} processed".format(i))

        i += 1
    except Exception as e:
        print(e.args)