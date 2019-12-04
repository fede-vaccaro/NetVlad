import os

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

from netvlad_model import NetVLADModel  # , NetVLADModelRetinaNet
from netvlad_model import input_shape



def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def create_image_dict(path):
    # input_shape = (224, 224, 3)
    img_list = get_imlist(path)
    tensor = {}
    for i, im_path in enumerate(img_list):
        img = image.load_img(im_path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img_key = im_path.strip(path)
        tensor[img_key] = img
        print("Img {} loaded".format(i))
    # tensor = np.array(tensor)
    return tensor


def main():
    print("Loading image dict")
    img_dict = create_image_dict('oxford5k/oxbuild_images/')

    my_model = NetVLADModel()
    my_model.build_netvladmodel()
    vgg_netvlad = my_model.get_netvlad_extractor()

    vgg_netvlad.load_weights("weights/model__pesi887.h5")

    all_key = list(img_dict.keys())

    img_tensor = [img_dict[key] for key in img_dict.keys()]
    img_tensor = np.array(img_tensor)

    print("Computing descriptors")
    all_feats = vgg_netvlad.predict(img_tensor, verbose=1)

    print("Computing NN")
    nbrs = NearestNeighbors(n_neighbors=len(all_key), metric='cosine').fit(all_feats)
    distances, indices = nbrs.kneighbors(all_feats)
    print(indices.shape)

if __name__ == "__main__":
    main()
