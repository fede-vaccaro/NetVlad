# Image Retrieval using Local CNN Features Pooling

#### Optional arguments to `$ python train.py`:
*  -h, --help            show this help message and exit
*  -m MODEL, --model MODEL
                        path to *specific* model checkpoint to load
*  -s START_EPOCH, --start-epoch START_EPOCH
                        epoch to restart training at
*  -c CONFIGURATION, --configuration CONFIGURATION
                        Yaml file where the configuration is stored
*  -t, --test            If must the training be bypassed for directly testing on Holidays
*  -k, --kmeans          If netvlad weights should be initialized for testing
*  -d DEVICE, --device DEVICE
                        CUDA device to be used. For info type `$ nvidia-smi`

#### Example of a configuration file 
```
# architecture
network:
  name: vgg # vgg or resnet
  output_layer: block5_conv2 # block5_conv2 for vgg or bn5c_branch2a/add_16 for resnet
  n_clusters: 64
  middle_pca:
    active: false
    dim: 512
    pretrain: true

# training
description: "vgg_test"
mining_batch_size: 2048
minibatch_size: 6
steps_per_epoch: 400
n_epochs: 160
lr:
  warm-up: true
  warm-up-steps: 2000 # if warm-up is active
  max_value: 1e-5
# min_value: max_value*0.1 by default

# testing
rotate_holidays: true
use_power_norm: true
use_multi_resolution: false
```
