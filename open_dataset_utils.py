import math
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
import torch
from PIL import Image
from PIL import ImageFile
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils import data

import utils
import netvlad_model as nm
import paths

ImageFile.LOAD_TRUNCATED_IMAGES = True


def show_triplet(triplet):
    fig = plt.figure(figsize=(10, 10))
    for i, t in enumerate(triplet):
        fig.add_subplot(1, 3, i + 1)
        t = t.astype('int')
        plt.imshow(t)

    plt.show()
    # plt.savefig("triplets/triplet{}.jpg".format(random.randint(1, 100000)))
    print("Triplet saved")
    time.sleep(5)


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocess_input(x):
    # RGB -> BGR
    x = x[:, :, [2, 1, 0]]
    mean = [103.939, 116.779, 123.68]

    x[:, :, 0] -= mean[0]
    x[:, :, 1] -= mean[1]
    x[:, :, 2] -= mean[2]

    return x


def restore(x):
    mean = [103.939, 116.779, 123.68]
    y = np.array(x)
    y[..., 0] += mean[0]
    y[..., 1] += mean[1]
    y[..., 2] += mean[2]

    # 'BGR'->'RGB'
    y = y[:, :, [2, 1, 0]]
    return y


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(u'.jpg')]


def load_img(path, transform):
    img = Image.open(path)
    img = transform(img)
    # img.resize(resize_shape, Image.BILINEAR)
    return img


# def open_img(path, input_shape=nm.NetVladBase.input_shape):
#     img = image.load_img(path, target_size=(input_shape[0], input_shape[1]), interpolation='bilinear')
#     img = image.img_to_array(img)
#     img = preprocess_input(img)
#     img_id = path.split('/')[-1]
#
#     return img, img_id

class TripletDataset(data.DataLoader):
    def __init__(self, triplet_list, transform):
        super(TripletDataset, self).__init__()
        self.transform = transform
        self.triplet_list = triplet_list
        print("Number of triplets generated: ", self.__len__())

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, index):
        im_a = load_img(path=self.triplet_list[index][0], transform=self.transform)
        im_p = load_img(path=self.triplet_list[index][1], transform=self.transform)
        im_n = load_img(path=self.triplet_list[index][2], transform=self.transform)
        return im_a, im_p, im_n


class ImagesFromListDataset(data.Dataset):
    def __init__(self, transform, image_list, label_list=None):
        super(ImagesFromListDataset, self).__init__()
        self.transform = transform

        self.set_image_list(image_list, label_list)

    def set_image_list(self, image_list, label_list):
        if label_list is not None:
            assert len(image_list) == len(label_list)
            self.label_list = label_list

        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im = load_img(path=self.image_list[index], transform=self.transform)
        if hasattr(self, 'label_list'):
            return im, self.label_list[index]
        else:
            return im


def torch_nn(feats, device, verbose=True):
    feats = torch.Tensor(feats).to(device)
    if verbose:
        print("Mining - Computing distances")
    distances = (feats.mm(feats.t())).cpu()
    del feats
    if verbose:
        print("Mining - Computing indices")
    distances, indices = distances.sort(descending=True)
    return distances, indices


class LandmarkTripletGenerator():
    def __init__(self, train_dir, model, transform, device, mining_batch_size=2048, minibatch_size=24, images_per_class=10,
                 semi_hard_prob=0.5, threshold=20, verbose=False, use_crop=False, print_statistics=False):

        self.print_statistics = print_statistics
        classes = os.listdir(train_dir)
        self.classes = classes
        self.train_dir = train_dir
        self.mining_batch_size = mining_batch_size
        self.images_per_class = images_per_class
        self.use_crop = use_crop
        self.select_neg = True

        # self.loader = Loader(batch_size=mining_batch_size, classes=classes, n_classes=n_classes, train_dir=train_dir,
        #                      transform=model.full_transform)

        self.transform = transform
        self.minibatch_size = minibatch_size
        self.model = model
        self.device = device
        self.verbose = verbose

        self.threshold = threshold
        self.semi_hard_prob = semi_hard_prob

        target_loss = 0.1
        delta = 0.02

        self.mining_iterations = 0

        self.loss_min = np.max(target_loss - delta, 0)
        self.loss_max = target_loss + delta

    def load_images_list(self, batch_size, classes, n_classes, train_dir):
        shuffled_classes = list(classes)
        random.shuffle(shuffled_classes)
        picked_classes = shuffled_classes[:n_classes]
        # load each image in those classes
        imgs = []
        for i, c in enumerate(picked_classes):
            images_in_c = os.listdir(train_dir + "/" + c)
            num_samples_in_c = len(images_in_c)
            random.shuffle(images_in_c)
            images_in_c = images_in_c[:min(batch_size // n_classes, num_samples_in_c)]
            for image_in_c in images_in_c:
                class_index = classes.index(c)
                imgs += [(image_in_c, class_index, c)]
        # randomize the image list
        random.shuffle(imgs)
        # pick the first batch_size (if enough)
        batch_size_ = min(batch_size, len(imgs))
        imgs = imgs[:batch_size_]
        images_list = []
        label_list = []
        # load the images
        # print("Opening the images (producer thread)")
        for im, label, dir in imgs:
            # image = load_img(path=train_dir + "/" + dir + "/" + im, transform=self.transform)
            # image = image.unsqueeze(0)
            images_list.append(train_dir + "/" + dir + "/" + im)
            label_list.append(label)

        return images_list, label_list

    def generator(self):
        while True:

            current_mining_batch_size = self.mining_batch_size
            current_images_per_class = self.images_per_class

            self.mining_iterations += 1
            print("\nMining - iteration {}; Mining batch size: {}; Images per class: {}".format(self.mining_iterations,
                                                                                                current_mining_batch_size,
                                                                                                current_images_per_class))

            image_list, label_list = self.load_images_list(batch_size=current_mining_batch_size, classes=self.classes,
                                                           n_classes=current_mining_batch_size // current_images_per_class,
                                                           train_dir=self.train_dir)
            if self.verbose:
                print("Mining - Computing descriptors")

            img_dataset = ImagesFromListDataset(image_list=image_list, label_list=label_list, transform=self.transform)
            b_size = 16
            data_loader = data.DataLoader(dataset=img_dataset, batch_size=b_size, num_workers=16, shuffle=False,
                                          pin_memory=True)

            n_step = math.ceil(len(img_dataset) / b_size)

            feats = utils.predict_generator_with_netlvad(generator=data_loader,
                                                         n_steps=n_step,
                                                         model=self.model,
                                                         device=self.device,
                                                         verbose=self.verbose)

            distances, indices = torch_nn(feats, device=self.device, verbose=self.verbose)

            # if self.verbose:
            #     print("Fitting NNs")
            # nbrs = NearestNeighbors(n_neighbors=10+self.threshold, metric='l2').fit(feats)
            # if self.verbose:
            #     print("Predicting NNs")
            # distances, indices = nbrs.kneighbors(feats)

            triplets = []
            losses = []

            sh_total_loss = []
            h_total_loss = []

            p_indices = []
            n_indices = []

            # find triplets:
            if self.verbose:
                print("Mining - Finding triplets")
            for i, row in enumerate(indices):
                anchor_label = label_list[i]

                j_neg = -1
                j_pos = -1

                for j, col in enumerate(row):
                    # find first negative
                    r_label = label_list[col]
                    if (j_pos == -1) and (j_neg == -1) and (
                            r_label == anchor_label):  # scorre finchè non trova il primo negativo
                        continue
                    elif (j_neg == -1) and (r_label != anchor_label):
                        j_neg = j
                        if j_neg > 1 and (np.random.uniform() > 1.0 - self.semi_hard_prob):
                            j_pos = j_neg - 1
                    elif (j_neg != -1) and (r_label == anchor_label):
                        j_pos = j

                    if (j_pos is not -1) and (j_neg is not -1) and (j_pos - j_neg < self.threshold):
                        if self.select_neg:
                            triplet = row[0], row[j_pos], row[j_neg], anchor_label, label_list[row[j_neg]]
                        else:
                            triplet = row[0], row[j_pos], row[j_neg], anchor_label, label_list[row[0]]

                        d_a_p_2 = np.max((2.0 - 2.0 * np.float64(distances[i][j_pos]), 0.0))
                        d_a_n_2 = np.max((2.0 - 2.0 * np.float64(distances[i][j_neg]), 0.0))

                        d_a_p = d_a_p_2
                        d_a_n = d_a_n_2

                        loss = 0.1 + d_a_p_2 - d_a_n_2

                        # print(loss)
                        if self.loss_min < loss < self.loss_max:
                            # print(loss)
                            triplets.append(triplet)
                            losses.append(loss)
                            p_indices += [j_pos]
                            n_indices += [j_neg]

                            if j_pos - j_neg > 0:  # hard triplet
                                h_total_loss += [loss]
                                if loss < 0.1 or (d_a_p < d_a_n):
                                    print(
                                        "Warning, hard triplet loss is less than 0.1! {}, indices: {}, {}".format(loss,
                                                                                                                  j_pos,
                                                                                                                  j_neg))
                            else:
                                sh_total_loss += [loss]
                                if loss > 0.1 or (d_a_p > d_a_n):
                                    print(
                                        "Warning, semi hard triplet loss is more than 0.1! {}, indices: {}, {}".format(
                                            loss, j_pos, j_neg))

                        break
                    elif j_neg != -1 and j_pos == -1:
                        if j - j_neg == self.threshold:
                            break

            class_set = []
            selected_triplets = []
            selected_losses = []

            del distances
            del indices
            # torch.cuda.empty_cache()

            for i, t in enumerate(triplets):
                anchor_label = t[3]
                if anchor_label in class_set:
                    continue
                else:
                    selected_triplets += [t]
                    selected_losses += [losses[i]]
                    class_set += [anchor_label]

            triplets = selected_triplets
            losses = selected_losses

            if False:
                print("Mining - Different classes: {}".format(len(class_set)))

            im_triplets = [[image_list[i], image_list[j], image_list[k]] for i, j, k, _, _ in triplets]
            im_labels = [[a, a, n] for _, _, _, a, n in triplets]

            # del image_list, indices, distances, feats
            # gc.collect()

            # select just K different classes
            # K_classes = 256
            n_triplets = len(im_triplets)
            K_classes = min(n_triplets, 256)
            im_triplets = im_triplets[:K_classes]

            anchors = [t[0] for t in im_triplets]
            positives = [t[1] for t in im_triplets]
            negatives = [t[2] for t in im_triplets]

            transform = self.model.train_transform if self.use_crop else self.transform
            img_a = ImagesFromListDataset(image_list=anchors,
                                          transform=transform)
            img_p = ImagesFromListDataset(image_list=positives,
                                          transform=transform)
            img_n = ImagesFromListDataset(image_list=negatives,
                                          transform=transform)

            data_loader_a = data.DataLoader(dataset=img_a, batch_size=self.minibatch_size, num_workers=4, shuffle=False,
                                            pin_memory=True)
            data_loader_p = data.DataLoader(dataset=img_p, batch_size=self.minibatch_size, num_workers=4, shuffle=False,
                                            pin_memory=True)
            data_loader_n = data.DataLoader(dataset=img_n, batch_size=self.minibatch_size, num_workers=4, shuffle=False,
                                            pin_memory=True)

            pages = math.ceil(K_classes / self.minibatch_size)
            print("Mining - Iterations available: {2}; {0} triplets, in batch of {1}".format(K_classes,
                                                                                             self.minibatch_size,
                                                                                             pages))
            if self.print_statistics:
                print("\nMining - statistics:")
                print("SH triplets: {0}; SH Mean Loss: {1:.4f}; SH Loss STD: {2:.4f}".format(len(sh_total_loss),
                                                                                             np.array(
                                                                                                 sh_total_loss).mean(),
                                                                                             np.array(
                                                                                                 sh_total_loss).std()))
                print("H triplets: {0}; H Mean Loss:{1:.4f}; H Loss STD {2:.4f}".format(len(h_total_loss),
                                                                                        np.array(h_total_loss).mean(),
                                                                                        np.array(h_total_loss).std()))
                print("Mean positive index: {}".format(np.array(p_indices).mean()))
                print("Mean negative index: {}".format(np.array(n_indices).mean()))

                h_mean_loss = np.array(h_total_loss).mean()
                sh_mean_loss = np.array(sh_total_loss).mean()

                sh_triplets = len(sh_total_loss)
                h_triplets = len(h_total_loss)

                sh_weight = sh_triplets / (sh_triplets + h_triplets)
                h_weight = h_triplets / (sh_triplets + h_triplets)

                predicted_loss = h_mean_loss * h_weight + sh_mean_loss * sh_weight

                print("Predicted training loss: {0:.5f}".format(predicted_loss))

            print("\n")
            for i, T in enumerate(zip(data_loader_a, data_loader_p, data_loader_n)):
                if i == pages:
                    continue
                yield T


def evaluation_triplet_generator(train_dir, netbatch_size=32, model=None):
    classes = os.listdir(train_dir)

    # n_classes = batch_size // 4
    picked_classes = classes

    # load each image in those classes
    imgs = []
    for i, c in enumerate(picked_classes):
        images_in_c = os.listdir(train_dir + "/" + c)
        # images_in_c = zip(images_in_c, [i]*len(images_in_c), [c]*len(images_in_c))
        for image_in_c in images_in_c:
            imgs += [(image_in_c, i, c)]

    # randomize the image list
    random.shuffle(imgs)

    # pick the first batch_size (if enough)

    images_array = []
    label_array = []

    # load the images
    print("Opening the images (evaluation)")
    for im, index, dir in imgs:
        image, _ = load_img(path=train_dir + "/" + dir + "/" + im, transform=model.full_transform)
        label = index
        images_array.append(image)
        label_array.append(label)

    images_array = np.array(images_array)
    label_array = np.array(label_array)

    while True:
        print("New iteration (evaluation)")
        # pick n_classes from the dirs
        # random.shuffle(classes)

        print("Computing descriptors (evaluation)")
        feats = model.predict(images_array)
        feats = normalize(feats)

        nbrs = NearestNeighbors(n_neighbors=len(images_array), metric='l2').fit(feats)
        distances, indices = nbrs.kneighbors(feats)

        triplets = []

        # find triplets:
        print("Finding triplets (evaluation)")
        for i, row in enumerate(indices):
            anchor_label = label_array[i]

            j_neg = -1
            j_pos = -1

            for j, col in enumerate(row):
                # find first negative
                r_label = label_array[col]
                if (j_neg == -1) and (
                        r_label == anchor_label):  # scorre finchè non trova il primo negativo
                    j_pos = j
                    # continue
                elif (j_neg == -1) and (r_label != anchor_label):
                    j_neg = j
                elif (j_neg != -1) and (r_label == anchor_label):  # se ha trovato prima un negativo di un positivo
                    j_pos = j

            if (j_pos is not -1) and (j_neg is not -1):
                triplet = row[0], row[j_pos], row[j_neg]
                triplets.append(triplet)

            if False:
                print("Distance between indices", j_pos - j_neg)
                print("L2 distance between query and positive: ", distances[i][j_pos])
                print("L2 distance between query and negative: ", distances[i][j_neg])
                print("Triplete Loss (a=0.1): ", 0.1 + distances[i][j_pos] ** 2. - distances[i][j_neg] ** 2.)
                # for t in triplet:
                #    plt.imshow(images_array[t]*0.5 + 0.5)
                #    plt.show()
                i, j, k = triplet
                t_ = (images_array[i], images_array[j], images_array[k])
                show_triplet(t_)
                # time.sleep(6)

        im_triplets = [[images_array[i], images_array[j], images_array[k]] for i, j, k in triplets]
        random.shuffle(im_triplets)
        DIM = min(len(images_array), 240)

        # del images_array, indices, distances, feats
        # gc.collect()

        # print(len(im_triplets))
        im_triplets = im_triplets[:DIM]

        pages = math.ceil(DIM / netbatch_size)

        keep_epoch = 1
        for e in range(keep_epoch):
            for page in range(pages):
                triplets_out = im_triplets[page * netbatch_size: min((page + 1) * netbatch_size, DIM)]

                anchors = torch.cat([t[0].unsqueeze(0) for t in triplets_out], dim=0)
                positives = torch.cat([t[1].unsqueeze(0) for t in triplets_out], dim=0)
                negatives = torch.cat([t[2].unsqueeze(0) for t in triplets_out], dim=0)

                yield [anchors, positives, negatives], None  # , [y_fake]*3

        # y_fake = np.zeros((len(images_array), net_output + 1))
        # yield [images_array, label_array], y_fake


import yaml


def main():
    conf_file = open('best-model-conf.yaml', 'r')
    conf = dict(yaml.safe_load(conf_file))
    conf_file.close()

    use_power_norm = conf['use_power_norm']
    use_multi_resolution = conf['use_multi_resolution']
    side_res = conf['input-shape']

    network_conf = conf['network']
    net_name = network_conf['name']

    nm.NetVladBase.input_shape = (side_res, side_res, 3)
    if use_multi_resolution:
        nm.NetVladBase.input_shape = (None, None, 3)

    my_model = None
    if net_name == "vgg":
        my_model = nm.NetVLADSiameseModel(**network_conf)
    elif net_name == "resnet":
        my_model = nm.VLADNet(**network_conf)
    else:
        print("Network name not valid.")

    vgg_netvlad = my_model.build_netvladmodel()
    weight_name = "best_model.h5"
    print("Loading weights: " + weight_name)
    vgg_netvlad.load_weights(weight_name)
    vgg_netvlad = my_model.get_netvlad_extractor()

    # vgg_netvlad = ResNet50(input_shape=(336,336,3), weights='imagenet', pooling='avg', include_top=False)

    landmark_generator = LandmarkTripletGenerator(train_dir=paths.landmark_clustered_path,
                                                  model=vgg_netvlad,
                                                  mining_batch_size=2048,
                                                  minibatch_size=6, semi_hard_prob=0.0,
                                                  threshold=5, verbose=True)

    train_generator = landmark_generator.generator()

    for el in train_generator:
        x, y = el
        a, p, n = x
        print("Number of triplets: ", a.shape[0])
        for i in range(a.shape[0]):
            print("Triplet loss: ", y[i])

            descs = vgg_netvlad.predict(np.array([a[i], p[i], n[i]]))
            descs = normalize(descs)
            a_desc = descs[0]
            p_desc = descs[1]
            n_desc = descs[2]

            alpha = 0.1
            d_a_p = np.linalg.norm(a_desc - p_desc, ord=2)
            d_a_n = np.linalg.norm(a_desc - n_desc, ord=2)

            loss = alpha + d_a_p - d_a_n

            print("Loss: ", loss)
            show_triplet([restore(a[i]), restore(p[i]), restore(n[i])])


if __name__ == '__main__':
    main()

# model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
# generator = LandmarkTripletGenerator("/mnt/m2/dataset/", model=model)
# custom_generator = generator.generator()
#
# show_triplets = True
# for el in custom_generator:
#     if show_triplets:
#         x = el[0]
#         a, p, n = x
#         print("Number of triplets: ", a.shape[0])
#         for i in range(a.shape[0]):
#             show_triplet([a[i], p[i], n[i]])
