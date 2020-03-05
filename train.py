# %%
import argparse
import os
import time
from math import ceil

import h5py
import holidays_testing_helpers as hth
import matplotlib.pyplot as plt
import netvlad_model
import numpy as np
import open_dataset_utils as my_utils
import paths
import utils
import yaml
from keras import backend as K
from keras import optimizers, Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import normalize
# from keras_radam import RAdam
# from keras_radam.training import RAdamOptimizer
from tqdm import tqdm
from triplet_loss import TripletL2LossLayerSoftmax

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
ap.add_argument("-c", "--configuration", type=str, default='train_configuration.yaml',
                help="Yaml file where the configuration is stored")
ap.add_argument("-t", "--test", action='store_true',
                help="If the training be bypassed for testing")
ap.add_argument("-k", "--kmeans", action='store_true',
                help="If netvlad weights should be initialized for testing")
ap.add_argument("-d", "--device", type=str, default="0",
                help="CUDA device to be used. For info type '$ nvidia-smi'")
args = vars(ap.parse_args())

test = args['test']
test_kmeans = args['kmeans']
model_name = args['model']
cuda_device = args['device']
config_file = args['configuration']

conf_file = open(config_file, 'r')
conf = dict(yaml.safe_load(conf_file))
conf_file.close()

# network
network_conf = conf['network']
net_name = network_conf['name']

n_cluster = network_conf['n_clusters']
middle_pca = network_conf['middle_pca']

# mining
try:
    threshold = conf['threshold']
except:
    threshold = 20

try:
    semi_hard_prob = conf['semi-hard-prob']
except:
    semi_hard_prob = 0.5

# training
train_description = conf['description']
mining_batch_size = conf['mining_batch_size']
minibatch_size = conf['minibatch_size']
steps_per_epoch = conf['steps_per_epoch']
epochs = conf['n_epochs']

# learning rate
lr_conf = conf['lr']
use_warm_up = lr_conf['warm-up']
warm_up_steps = lr_conf['warm-up-steps']
max_lr = float(lr_conf['max_value'])

# testing
rotate_holidays = conf['rotate_holidays']
use_power_norm = conf['use_power_norm']
use_multi_resolution = conf['use_multi_resolution']

side_res = conf['input-shape']

netvlad_model.NetVladBase.input_shape = (side_res, side_res, 3)
if use_multi_resolution:
    netvlad_model.NetVladBase.input_shape = (None, None, 3)

# if test:
#     gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#     for device in gpu_devices:
#         tf.config.experimental.set_memory_growth(device, True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

my_model = None
if net_name == "vgg":
    my_model = netvlad_model.NetVLADSiameseModel(**network_conf)
elif net_name == "resnet":
    # my_model = netvlad_model.NetVladResnet(**network_conf)
    my_model = netvlad_model.GeMResnet(**network_conf)
else:
    print("Network name not valid.")

vgg, output_shape = my_model.get_feature_extractor(verbose=False)

vgg_netvlad = my_model.build_netvladmodel()

print("Netvlad output shape: ", vgg_netvlad.output_shape)
print("Feature extractor output shape: ", vgg.output_shape)

train_pca = False
train_kmeans = (not test or test_kmeans) and model_name is None and not train_pca
train = not test

if False:
    init_generator = image.ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        paths.landmarks_path,
        target_size=(netvlad_model.NetVladBase.input_shape[0], netvlad_model.NetVladBase.input_shape[1]),
        batch_size=128 // 4,
        class_mode=None,
        interpolation='bilinear', seed=4242)

    my_model.train_kmeans(init_generator)

    if network_conf['post_pca']['active']:
        my_model.pretrain_pca(init_generator)

preload_means = True

# initialize softmax
if train_kmeans:
    means = {}

    init_generator = image.ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        paths.landmark_clustered_path,
        target_size=(netvlad_model.NetVladBase.input_shape[0], netvlad_model.NetVladBase.input_shape[1]),
        batch_size=minibatch_size,
        class_mode=None,
        interpolation='bilinear', shuffle=False)
    class_indices = init_generator.class_indices

    if not preload_means:

        clusters = {}

        img_list = [os.path.splitext(os.path.split(f)[1])[0] for f in init_generator.filenames]
        n_steps = int(np.ceil(len(img_list) / minibatch_size))

        all_feats = my_model.get_netvlad_extractor().predict_generator(init_generator, steps=n_steps, verbose=1)
        labels = init_generator.classes[:min(minibatch_size * n_steps, len(img_list))]

        indices_classes = {}
        for class_ in class_indices.keys():
            index = init_generator.class_indices[class_]
            indices_classes[str(index)] = class_

            output_shape = all_feats.shape[1]

        for i in set(init_generator.labels):
            clusters[indices_classes[str(i)]] = []
            # means[indices_classes[str(i)]] = []

        print("Preparing clusters")
        for i, z in enumerate(zip(all_feats, labels)):
            feat, y = z
            clusters[indices_classes[str(y)]] += [feat]

        print("Preparing means")
        means_h5 = h5py.File('means.h5', 'w')
        for label in clusters.keys():
            # descs_array = normalize(np.array(clusters[label]))
            descs_array = np.array(clusters[label])
            samples = len(descs_array)
            mean = np.sum(descs_array, axis=0)
            # mean /= np.linalg.norm(mean, ord=2)
            mean /= samples
            means[label] = mean
            means_h5.create_dataset(name=str(label), data=mean)

        means_h5.close()
    else:
        means_h5 = h5py.File('means.h5', 'r')
        for key in means_h5.keys():
            means[key] = means_h5[key][:]
        n_classes = len(means.keys())
        means_h5.close()

    print("Initializing matrix for softmax")

    output_shape = 2048

    centroids = np.zeros((len(means.keys()), output_shape))

    classes = os.listdir(paths.landmark_clustered_path)
    for label in means.keys():
        index = class_indices[label]
        index = int(index)
        mean = means[label]
        centroids[index] = mean

if train:
    steps_per_epoch = steps_per_epoch
    vgg_netvlad.summary()

    start_epoch = int(args['start_epoch'])
    triplet_loss_layer = TripletL2LossLayerSoftmax(n_classes=len(os.listdir(paths.landmark_clustered_path)), alpha=0.1,
                                                   l=0.5)
    vgg_netvlad = Model(vgg_netvlad.input, triplet_loss_layer(vgg_netvlad.output))

    if model_name is not None:
        print("Resuming training from epoch {} at iteration {}".format(start_epoch, steps_per_epoch * start_epoch))
        vgg_netvlad.load_weights(model_name)
        # vgg_netvlad.summary()
    else:
        # set triplet loss layer softmax weights
        weights = triplet_loss_layer.get_weights()

        centroids = normalize(centroids)

        alpha = 0.1
        assignments_weights = 2. * alpha * centroids
        assignments_bias = -alpha * np.sum(np.power(centroids, 2), axis=1)

        centroids = centroids.T
        assignments_weights = assignments_weights.T
        assignments_bias = assignments_bias.T

        weights_softmax = triplet_loss_layer.get_weights()

        weights_softmax[0] = assignments_weights
        weights_softmax[1] = assignments_bias

        triplet_loss_layer.set_weights(weights_softmax)

    opt = optimizers.Adam(lr=1e-5)
    vgg_netvlad.compile(opt)

    steps_per_epoch_val = ceil(1491
                               / minibatch_size)

    init_generator = my_utils.LandmarkTripletGenerator(train_dir=paths.landmark_clustered_path,
                                                       model=my_model.get_netvlad_extractor(),
                                                       mining_batch_size=mining_batch_size,
                                                       minibatch_size=minibatch_size, semi_hard_prob=semi_hard_prob,
                                                       threshold=threshold, use_positives_augmentation=False,
                                                       class_indices=class_indices)

    train_generator = init_generator.generator()

    test_generator = my_utils.evaluation_triplet_generator(paths.holidays_small_labeled_path,
                                                           model=my_model.get_netvlad_extractor(),
                                                           netbatch_size=minibatch_size)

    losses = []
    # val_losses = []
    val_maps = []

    not_improving_counter = 0
    not_improving_thresh = 15

    description = train_description

    # val_loss_e = []
    #
    # for s in range(steps_per_epoch_val):
    #     x_val, _ = next(test_generator)
    #     val_loss_s = vgg_netvlad.predict_on_batch(x_val)
    #     val_loss_e.append(val_loss_s)

    # starting_val_loss = np.array(val_loss_e).mean()
    # print("Starting validation loss: ", starting_val_loss)

    starting_map = hth.tester.test_holidays(model=my_model.get_netvlad_extractor(), side_res=side_res,
                                            use_multi_resolution=use_multi_resolution,
                                            rotate_holidays=rotate_holidays, use_power_norm=use_power_norm,
                                            verbose=False)

    print("Starting mAP: ", starting_map)

    for e in range(epochs):
        t0 = time.time()

        losses_e = []

        pbar = tqdm(range(steps_per_epoch))

        for s in pbar:
            it = K.get_value(vgg_netvlad.optimizer.iterations)
            if use_warm_up:
                lr = utils.lr_warmup(it, wu_steps=2000, min_lr=1.e-6, max_lr=1.e-5, exp_decay=False,
                                     exp_decay_factor=(0.1) / (80 * 400))
            else:
                lr = max_lr

            K.set_value(vgg_netvlad.optimizer.lr, lr)

            x, y = next(train_generator)
            # print("Starting training at epoch ", e)
            loss_s = vgg_netvlad.train_on_batch(x + y, None)
            losses_e.append(loss_s)
            description_tqdm = "Loss at epoch {0}/{3} step {1}: {2:.4f}. Lr: {4}".format(e + start_epoch, s, loss_s,
                                                                                         epochs + start_epoch, lr)
            pbar.set_description(description_tqdm)

        print("")
        loss = np.array(losses_e).mean()
        losses.append(loss)

        # val_loss_e = []

        # for s in range(steps_per_epoch_val):
        #     x_val, _ = next(test_generator)
        #     val_loss_s = vgg_netvlad.predict_on_batch(x_val)
        #     val_loss_e.append(val_loss_s)

        # val_loss = np.array(val_loss_e).mean()
        val_map = hth.tester.test_holidays(model=my_model.get_netvlad_extractor(), side_res=side_res,
                                           use_multi_resolution=use_multi_resolution,
                                           rotate_holidays=rotate_holidays, use_power_norm=use_power_norm,
                                           verbose=False)

        max_val_map = starting_map

        if e > 0:
            max_val_map = np.max(val_maps)
        else:
            # val_losses.append(min_val_loss)
            val_maps.append(max_val_map)

        val_maps.append(val_map)

        # if val_loss < min_val_loss:
        #     model_name = "model_e{0}_{2}_{1:.4f}.h5".format(e + start_epoch, val_loss, description)
        #     print("Val. loss improved from {0:.4f}. Saving model to: {1}".format(min_val_loss, model_name))
        #     vgg_netvlad.save_weights(model_name)
        #     not_improving_counter = 0
        if val_map > max_val_map:
            model_name = "model_e{0}_{2}_{1:.4f}.h5".format(e + start_epoch, val_map, description)
            print("Val. loss improved from {0:.4f}. Saving model to: {1}".format(max_val_map, model_name))
            vgg_netvlad.save_weights(model_name)
            not_improving_counter = 0
        else:
            print("Val loss ({0:.4f}) did not improve from {1:.4f}".format(val_map, max_val_map))
            not_improving_counter += 1
            print("Val loss does not improve since {} epochs".format(not_improving_counter))
            if e % 5 == 0:
                # model_name = "model_e{0}_{2}_{1:.4f}_checkpoint.h5".format(e + start_epoch, val_loss, description)
                model_name = "model_e{0}_{2}_{1:.4f}_checkpoint.h5".format(e + start_epoch, val_map, description)
                vgg_netvlad.save_weights(model_name)
                print("Saving model to: {} (checkpoint)".format(model_name))
            # if not_improving_counter == not_improving_thresh:
            if False:
                # lr *= 0.5
                # K.set_value(vgg_netvlad.optimizer.lr, lr)
                # print("Learning rate set to: {}".format(lr))
                opt = optimizers.Adam(lr=lr)
                vgg_netvlad.compile(opt)
                not_improving_counter = 0
                print("Optimizer weights restarted.")

        print("Validation mAP: {}\n".format(val_map))
        # print("Validation loss: {}\n".format(val_loss))
        print("Training loss: {}\n".format(loss))

        t1 = time.time()
        print("Time for epoch {}: {}s".format(e, int(t1 - t0)))

    init_generator.loader.stop_loading()

    model_name = "model_e{}_{}_.h5".format(epochs + start_epoch, description)
    vgg_netvlad.save_weights(model_name)
    print("Saved model to disk: ", model_name)

    plt.figure(figsize=(8, 8))
    # plt.plot(val_maps, label='validation map')
    plt.plot(losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.savefig("train_val_loss_{}.pdf".format(description))

print("Testing model")
print("Input shape: ", netvlad_model.NetVladBase.input_shape)

if test and model_name is not None:
    print("Loading ", model_name)
    vgg_netvlad.load_weights(model_name)

vgg_netvlad = my_model.get_netvlad_extractor()

hth.test_holidays(model=vgg_netvlad, side_res=side_res, use_multi_resolution=use_multi_resolution,
                  rotate_holidays=rotate_holidays, use_power_norm=use_power_norm, verbose=True)
