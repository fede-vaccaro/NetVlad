import pandas as pd

lm = pd.read_csv('landmarks.csv', index_col='landmark_id')

lm_dict = lm.T.to_dict('list')

n_landmarks = len(lm_dict)

import shutil

import os

def impath(id):
    path = id[0] + "/" + id[1] + "/" + id[2] + "/"
    path += id + ".jpg"
    return path

i = 0
for landmark_id in list(lm_dict.keys())[:10]:
    loc = "ld/"
    if not os.path.exists(loc + str(landmark_id)):
        os.makedirs(loc + str(landmark_id))

    im_list = lm_dict[landmark_id][0]
    im_list = im_list.split()

    im_list = [(impath(x), x) for x in im_list]
    #print(im_list)
    for path, img in im_list:
        dest = loc + str(landmark_id) + "/" + img + ".jpg"
    #    print(dest)
        try:
            shutil.copy(path, dest)
        except:
            pass
    i += 1
    print("Landmark {}/{}".format(i, n_landmarks))