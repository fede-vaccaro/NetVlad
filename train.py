# %%
import argparse
import os
import time
from itertools import cycle
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision.datasets import folder
from tqdm import tqdm

import holidays_testing_helpers as hth
import netvlad_model
import open_dataset_utils as my_utils
import paths
import utils
from evaluate_dataset import compute_aps
from torch_triplet_loss import TripletLoss

ap = argparse.ArgumentParser()

ap.add_argument("-e", "--export", type=str, default="",
                help="Dir where to export checkpoints")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-c", "--configuration", type=str, default='resnet-conf.yaml',
                help="Yaml file where the configuration is stored")
ap.add_argument("-t", "--test", action='store_true',
                help="If the training be bypassed for testing")
ap.add_argument("-val", "--validation", action='store_true', default=False,
                help="Computing loss on validation test")
ap.add_argument("-k", "--kmeans", action='store_true',
                help="If netvlad weights should be initialized for testing")
ap.add_argument("-d", "--device", type=str, default="0",
                help="CUDA device to be used. For info type '$ nvidia-smi'")
ap.add_argument("-v", "--verbose", action='store_true', default=False,
                help="Verbosity mode.")

args = vars(ap.parse_args())

test = args['test']
test_kmeans = args['kmeans']
model_name = args['model']
cuda_device = args['device']
config_file = args['configuration']
compute_validation = args['validation']
verbose = args['verbose']

EXPORT_DIR = args['export']

conf_file = open(config_file, 'r')
conf = dict(yaml.safe_load(conf_file))
conf_file.close()

print("Loaded configuration: ")
for key in conf.keys():
    print(" - {}: {}".format(key, conf[key]))

# network
network_conf = conf['network']

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
mining_batch_size = conf['mining_batch_size']
images_per_class = conf['images_per_class']

mining_batch_size = [int(mbs) for mbs in str(mining_batch_size).split(",")]
images_per_class = [int(ipc) for ipc in str(images_per_class).split(",")]

# training
train_description = conf['description']
minibatch_size = conf['minibatch_size']
steps_per_epoch = conf['steps_per_epoch']
epochs = conf['n_epochs']
use_crop = conf['use_crop']
checkpoint_freq = conf['checkpoint_freq']
memory_saving = conf['memory_saving']

# learning rate
lr_conf = conf['lr']
use_warm_up = lr_conf['warm-up']
warm_up_steps = lr_conf['warm-up-steps']
max_lr = float(lr_conf['max_value'])
min_lr = float(lr_conf['min_value'])
lr_decay = float(lr_conf['lr_decay'])
step_frequency = lr_conf['step_frequency']

# testing
rotate_holidays = conf['rotate_holidays']
use_power_norm = conf['use_power_norm']
use_multi_resolution = conf['use_multi_resolution']

side_res = conf['input-shape']

netvlad_model.NetVladBase.input_shape = (side_res, side_res, 3)

# if test:
#     gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#     for device in gpu_devices:
#         tf.config.experimental.set_memory_growth(device, True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vladnet = netvlad_model.VLADNet(**network_conf)

transform = vladnet.full_transform


# vgg, output_shape = vladnet.get_feature_extractor(verbose=False)
# vgg_netvlad = vladnet.build_netvladmodel()
# vgg_netvlad.summary()

# print("Netvlad output shape: ", vgg_netvlad.output_shape)
# print("Feature extractor output shape: ", vgg.output_shape)

train_pca = False
train_kmeans = (not test or test_kmeans) and model_name is None and not train_pca and (
        network_conf['pooling_type'] != 'gem')
train = not test

if train_kmeans:
    image_folder = folder.ImageFolder(root=paths.landmarks_path, transform=transform)

    # init_generator = image.ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    #     paths.landmarks_path,
    #     target_size=(netvlad_model.NetVladBase.input_shape[0], netvlad_model.NetVladBase.input_shape[1]),
    #     batch_size=128 // 4,
    #     class_mode=None,
    #     interpolation='bilinear', seed=4242)
    # Print model's state_dict
    if verbose:
        print("Model's state_dict:")
        for param_tensor in vladnet.state_dict():
            print(param_tensor, "\t", vladnet.state_dict()[param_tensor].size())
    if network_conf['middle_pca']['active'] and network_conf['middle_pca']['pretrain']:
        vladnet.initialize_whitening(image_folder)

    vladnet.initialize_netvlad(image_folder)

    if network_conf['post_pca']['active']:
        vladnet.initialize_whitening(image_folder)

if torch.cuda.device_count() > 1:
    print("Available GPUS: ", torch.cuda.device_count())
    vladnet = torch.nn.DataParallel(vladnet, device_ids=range(torch.cuda.device_count()))

if train:
    vladnet.to(device)

    steps_per_epoch = steps_per_epoch

    start_epoch = 0

    # define opt
    adam = opt = torch.optim.Adam(lr=max_lr, params=vladnet.parameters())

    if use_warm_up:
        lr_lambda = utils.lr_warmup(wu_steps=warm_up_steps, min_lr=min_lr / max_lr, max_lr=1.0,
                                    frequency=step_frequency * steps_per_epoch,
                                    step_factor=0.1, weight_decay=lr_decay)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adam, lr_lambda=[lr_lambda])

    # TODO reload model
    if model_name is not None:
        # vladnet.load_weights(model_name)
        # vgg_netvlad.summary()
        checkpoint = torch.load(model_name)
        vladnet.load_state_dict(checkpoint['model_state_dict'])
        adam.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming training from epoch {} at iteration {}".format(start_epoch, steps_per_epoch * start_epoch))

    # define loss
    criterion = TripletLoss()

    steps_per_epoch_val = ceil(1491
                               / minibatch_size)

    # load generators
    landmarks_triplet_generator = my_utils.LandmarkTripletGenerator(train_dir=paths.landmarks_path,
                                                                    model=vladnet,
                                                                    transform=transform,
                                                                    device=device,
                                                                    mining_batch_size=mining_batch_size[0],
                                                                    images_per_class=images_per_class[0],
                                                                    minibatch_size=minibatch_size,
                                                                    semi_hard_prob=semi_hard_prob,
                                                                    threshold=threshold, verbose=True,
                                                                    use_crop=use_crop)

    train_generator = landmarks_triplet_generator.generator()

    # TODO fix
    if compute_validation:
        test_generator = my_utils.evaluation_triplet_generator(paths.holidays_pic_path,
                                                               model=vladnet,
                                                               netbatch_size=minibatch_size)

    losses = []
    val_losses = []
    val_maps = []

    not_improving_counter = 0

    description = train_description

    val_loss_e = []

    if compute_validation:
        for s in range(steps_per_epoch_val):
            x_val, _ = next(test_generator)
            val_loss_s = vladnet.predict_with_netvlad(x_val)
            val_loss_e.append(val_loss_s)

        starting_val_loss = np.array(val_loss_e).mean()
        print("Starting validation loss: ", starting_val_loss)

    # print("Starting Oxford5K mAP: ", np.array(compute_aps(dataset='o', model=vladnet, verbose=True)).mean())
    # print("Starting Paris6K mAP: ", np.array(compute_aps(dataset='p', model=vladnet, verbose=True)).mean())
    starting_map = hth.tester.test_holidays(model=vladnet,
                                            device=device,
                                            transform=transform,
                                            side_res=side_res,
                                            use_multi_resolution=use_multi_resolution,
                                            rotate_holidays=rotate_holidays, use_power_norm=use_power_norm,
                                            verbose=True)

    print("Starting mAP: ", starting_map)

    for e in range(epochs - start_epoch):
        print("Remaining epochs: ", epochs - start_epoch)
        t0 = time.time()

        losses_e = []

        pbar = tqdm(range(steps_per_epoch))

        landmarks_triplet_generator.images_per_class = images_per_class[(e + start_epoch) % len(images_per_class)]
        landmarks_triplet_generator.mining_batch_size = mining_batch_size[(e + start_epoch) % len(mining_batch_size)]

        for s in pbar:
            a, p, n = next(train_generator)
            vladnet.eval()

            # clear gradient
            adam.zero_grad()

            batch_size = a.shape[0]
            batch = torch.cat([a, p, n], dim=0).to(device)
            # print("Batch shape: ", batch.shape)
            # print("Batch shape a, p, n : ", a.shape, p.shape, n.shape)

            # loss
            if not memory_saving:
                features = vladnet.forward(batch)
                d_a = features[0:batch_size]
                d_p = features[batch_size:batch_size*2]
                d_n = features[batch_size*2:batch_size*3]

                loss_s = criterion(d_a, d_p, d_n)
                loss_s.backward()
            else:
                loss_s = 0.0
                for a, p, n in zip(a, p, n):
                    a_d = vladnet.forward(a.unsqueeze(0).to(device))
                    p_d = vladnet.forward(p.unsqueeze(0).to(device))
                    n_d = vladnet.forward(n.unsqueeze(0).to(device))

                    loss_mini_step = criterion(a_d, p_d, n_d) / minibatch_size
                    loss_mini_step.backward()
                    loss_s += float(loss_mini_step)

            adam.step()
            if use_warm_up:
                lr_scheduler.step(epoch=(e + start_epoch) * steps_per_epoch + s)

            losses_e.append(float(loss_s))

            it = s + (e + start_epoch) * steps_per_epoch
            if use_warm_up:
                lr = lr_lambda(it)
            else:
                lr = max_lr
            description_tqdm = "Loss at epoch {0}/{3} step {1}: {2:.4f}. Lr: {4}".format(e + start_epoch, s, loss_s,
                                                                                         epochs, lr)
            pbar.set_description(description_tqdm)

        print("")

        loss = np.array(losses_e).mean()
        losses.append(loss)

        val_loss_e = []

        if compute_validation:
            for s in range(steps_per_epoch_val):
                x_val, _ = next(test_generator)
                val_loss_s = vladnet.predict_with_netvlad(x_val)
                val_loss_e.append(val_loss_s)

            val_loss = np.array(val_loss_e).mean()
        val_map = hth.tester.test_holidays(model=vladnet,
                                           device=device,
                                           transform=transform,
                                           side_res=side_res,
                                           use_multi_resolution=use_multi_resolution,
                                           rotate_holidays=rotate_holidays, use_power_norm=use_power_norm,
                                           verbose=False)

        max_val_map = starting_map
        if compute_validation:
            min_val_loss = starting_val_loss

        if e > 0:
            max_val_map = np.max(val_maps)
            if compute_validation:
                min_val_loss = np.min(val_losses)
        else:
            val_maps.append(max_val_map)
            if compute_validation:
                val_losses.append(min_val_loss)

        val_maps.append(val_map)
        if compute_validation:
            val_losses.append(val_loss)

        # if val_loss < min_val_loss:
        #     model_name = "model_e{0}_{2}_{1:.4f}.h5".format(e + start_epoch, val_loss, description)
        #     print("Val. loss improved from {0:.4f}. Saving model to: {1}".format(min_val_loss, model_name))
        #     vgg_netvlad.save_weights(model_name)
        #     not_improving_counter = 0
        # if val_map > max_val_map:
        if False:
            model_name = "model_e{0}_{2}_{1:.4f}.pkl".format(e + start_epoch, val_map, description)
            model_name = os.path.join(EXPORT_DIR, model_name)
            print("Val. mAP improved from {0:.4f}".format(max_val_map, model_name))
            # torch.save({
            #     'epoch': e + start_epoch,
            #     'model_state_dict': vladnet.state_dict(),
            #     'optimizer_state_dict': adam.state_dict(),
            # }, model_name)
            not_improving_counter = 0
        else:
            print("Val mAP ({0:.4f}) did not improve from {1:.4f}".format(val_map, max_val_map))
            not_improving_counter += 1
            print("Val mAP does not improve since {} epochs".format(not_improving_counter))
            if (e + start_epoch) % checkpoint_freq == 0:
                model_name = "model_e{0}_{2}_{1:.4f}_checkpoint.pkl".format(e + start_epoch, val_map, description)
                model_name = os.path.join(EXPORT_DIR, model_name)
                torch.save({
                    'epoch': e + start_epoch,
                    'model_state_dict': vladnet.state_dict(),
                    'optimizer_state_dict': adam.state_dict(),
                }, model_name)
                print("Saving model to: {} (checkpoint)".format(model_name))

        print("Validation mAP: {}\n".format(val_map))
        print("Oxford5K mAP: ",
              np.array(compute_aps(dataset='o', model=vladnet, verbose=True, transform=transform)).mean())
        # print("Paris 6K mAP: ",
        #       np.array(compute_aps(dataset='p', model=vladnet, verbose=True)).mean())

        if compute_validation:
            print("Validation loss: {}\n".format(val_loss))
        print("Training loss: {}\n".format(loss))

        t1 = time.time()
        print("Time for epoch {}: {}s".format(e, int(t1 - t0)))

    landmarks_triplet_generator.loader.stop_loading()

    model_name = "model_e{}_{}_.pkl".format(epochs, description)
    model_name = os.path.join(EXPORT_DIR, model_name)
    torch.save({
        'epoch': epochs,
        'model_state_dict': vladnet.state_dict(),
        'optimizer_state_dict': adam.state_dict(),
    }, model_name)
    print("Saved model to disk: ", model_name)

    plt.figure(figsize=(8, 8))
    plt.plot(np.array(val_maps), label='validation map')
    plt.plot(losses, label='training loss')
    if compute_validation:
        plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.savefig("train_val_loss_{}.pdf".format(description))

print("Testing model")
hth.tester.test_holidays(model=vladnet, side_res=side_res, use_multi_resolution=use_multi_resolution,
                         rotate_holidays=rotate_holidays, use_power_norm=use_power_norm, verbose=True)
