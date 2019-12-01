import os, shutil
path = "holidays_small_/"
hd = os.listdir(path)

for dir in hd:
    imgs = os.listdir(path + dir)
    print(imgs)
    for im in imgs:
        shutil.copy(path + dir + "/" + im, "holidays_small")