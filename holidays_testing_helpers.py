import math
import os

import PIL
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
# from keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import torch
import paths
import utils
from netvlad_model import NetVladBase


def preprocess_input(x):
    # RGB -> BGR
    x = x[:, :, [2, 1, 0]]
    mean = [103.939, 116.779, 123.68]

    x[:, :, 0] -= mean[0]
    x[:, :, 1] -= mean[1]
    x[:, :, 2] -= mean[2]

    return x


def get_imlist_(path="holidays_small_2"):
    imnames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]
    imnames = [path.strip("holidays_small_2/") for path in imnames]
    imnames = [path.strip('.jpg') for path in imnames]
    return imnames


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def create_image_dict(img_list, input_shape, preprocess_input=preprocess_input, rotate=False):
    # input_shape = (224, 224, 3)
    tensor = {}

    rotated = open('holidays-rotate.yaml', 'r')
    rotated_imgs = dict(yaml.safe_load(rotated))
    rotated.close()

    for path in img_list:
        # img = image.load_img(path, target_size=(input_shape[0], input_shape[1]), interpolation='bilinear')
        img = Image.open(path)
        img.resize((input_shape[0], input_shape[1]), Image.BILINEAR)
        img_key = path.strip(paths.holidays_pic_path)

        if img_key in list(rotated_imgs.keys()) and rotate:
            # print(img_key)
            degrees = rotated_imgs[img_key]
            degrees = int(degrees)
            img = img.rotate(-degrees)

        # img = np.asarray(img)
        img = preprocess_input(img)
        tensor[img_key] = img.unsqueeze(0)
    # tensor = np.array(tensor)

    img_tensor = [tensor[key] for key in tensor]
    #img_tensor = np.array(img_tensor)
    img_tensor = torch.cat(img_tensor, dim=0)
    return img_tensor


def make_perfect_holidays_result(imnames, q_ids):
    perfect_idx = []
    for qimno in q_ids:
        qname = imnames[qimno]
        positive_results = set([i for i, name in enumerate(imnames) if name != qname and name[:4] == qname[:4]])
        ok = [qimno] + [i for i in positive_results]
        others = [i for i in range(1491) if i not in positive_results and i != qimno]
        perfect_idx.append(ok + others)
    return np.array(perfect_idx)


def mAP(q_ids, idx, imnames):
    aps = []
    for qimno, qres in zip(q_ids, idx):
        qname = imnames[qimno]
        # collect the positive results in the dataset
        # the positives have the same prefix as the query image
        positive_results = set([i for i, name in enumerate(imnames)
                                if name != qname and name[:4] == qname[:4]])
        #
        # ranks of positives. We skip the result #0, assumed to be the query image
        ranks = [i for i, res in enumerate(qres[1:]) if res in positive_results]
        #
        # accumulate trapezoids with this basis
        recall_step = 1.0 / len(positive_results)
        ap = 0
        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far
            # y-size on left side of trapezoid:
            precision_0 = ntp / float(rank) if rank > 0 else 1.0
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        # print('query %s, AP = %.3f' % (qname, ap))
        aps.append(ap)
    return np.mean(aps)


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


def show_result(display_idx, query_imids, imnames, nqueries=10, nresults=10, ts=(100, 100)):
    if nqueries is not None:
        nrow = nqueries  # number of query images to show

    if nresults is not None:
        nres = 10  # number of results per query

    for qno in range(nrow):
        imfiles = []
        oks = [True]
        # show query image with white outline
        qimno = query_imids[qno]
        imfiles.append('holidays_small/' + imnames[qimno] + '.jpg')
        for qres in display_idx[qno, :nres]:
            # use image name to determine if it is a TP or FP result
            oks.append(imnames[qres][:4] == imnames[qimno][:4])
            imfiles.append('holidays_small/' + imnames[qres] + '.jpg')
        # print(qno, (imfiles))
        plt.imshow(montage(imfiles, thumb_size=ts, ok=oks, shape=(1, nres)))
        plt.show()


class HolidaysTester:
    img_tensor = None

    def test_holidays(self, side_res, model: NetVladBase, use_power_norm=False, use_multi_resolution=False, rotate_holidays=True,
                      verbose=False):
        imnames = get_imlist_()
        query_imids = [i for i, name in enumerate(imnames) if name[-2:].split('.')[0] == "00"]
        if verbose:
            print('tot images = %d, query images = %d' % (len(imnames), len(query_imids)))
        base_resolution = (side_res, side_res, 3)
        input_shape_1 = (768, 768, 3)
        input_shape_2 = (504, 504, 3)
        input_shape_3 = (224, 224, 3)
        input_shapes = [input_shape_2, input_shape_3]
        if verbose:
            print("Loading images")
        if self.img_tensor is None:
            self.img_tensor = create_image_dict(get_imlist(paths.holidays_pic_path), input_shape=base_resolution,
                                           rotate=rotate_holidays, preprocess_input=model.full_transform)
        else:
            print("Using preallocated image tensor")

        if verbose:
            print("Extracting features")

        verbose_ = 0
        if verbose:
            verbose_ = 1
        # all_feats = model.predict(self.img_tensor, verbose=verbose_, batch_size=3)
        all_feats = model.predict_with_netvlad(img_tensor=self.img_tensor)
        if use_multi_resolution:
            for shape in input_shapes:
                img_tensor = create_image_dict(get_imlist(paths.holidays_pic_path), input_shape=shape,
                                               rotate=True, preprocess_input=model.full_transform)
                batch_size = 32
                if shape[0] >= 768:
                    batch_size = 12

                all_feats += model.predict_with_netvlad(img_tensor=img_tensor, batch_size=batch_size)
        all_feats = normalize(all_feats)
        use_pca = False
        if use_pca:
            n_components = 2048

            pca_dataset = h5py.File("pca_{}.h5".format(n_components), 'r')
            mean = pca_dataset['mean'][:]
            components = pca_dataset['components'][:]
            explained_variance = pca_dataset['explained_variance'][:]
            pca_dataset.close()

            all_feats = utils.transform(all_feats, mean, components, explained_variance, whiten=True, pow_whiten=0.5)

        if use_power_norm:
            all_feats_sign = np.sign(all_feats)
            all_feats = np.power(np.abs(all_feats), 0.5)
            all_feats = np.multiply(all_feats, all_feats_sign)

        query_feats = all_feats[query_imids]
        nbrs = NearestNeighbors(n_neighbors=1491, metric='cosine').fit(all_feats)
        distances, indices = nbrs.kneighbors(query_feats)
        meanAP = mAP(query_imids, indices, imnames=imnames)
        if verbose:
            print('mean AP = %.3f' % meanAP)
        return meanAP


tester = HolidaysTester()
