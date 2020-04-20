# Image Retrieval using Local CNN Features Pooling

#### Needed package:
* pytorch
* torchvision
* SciKit Learn
* NumPy
* MatPlotLib
* Pillow = 6.0.0

#### First you need to create the file `paths.py` with:
```
landmarks_path = PATH_TO_DATASET
# deprecated # holidays_small_labeled_path = ""
# deprecated # oxford_small_labeled_path = ""
holidays_pic_path = PATH_TO_HOLIDAYS
path_oxford = PATH_TO_OXFORD5K_DATASET
path_paris = PATH_TO_PARIS6K_DATASET
```

#### For training the inference model `$ python train.py`:
*  -h, --help            show this help message and exit
*  -m MODEL, --model MODEL
                        path to *specific* model checkpoint to load for resuming the training;
*  -c CONFIGURATION, --configuration CONFIGURATION
                        Yaml file where the configuration is stored
*  -t, --test            If must the training be bypassed for directly testing on Holidays
*  -k, --kmeans          If netvlad weights should be initialized for testing
*  -d DEVICE, --device DEVICE
                        CUDA device to be used. For info type `$ nvidia-smi`
* -e EXPORT_DIR, --export EXPORT_DIR
                        Location where to export the models
#### for training the PCA Whitening `$ python train_pca.py`
This script will extract a set of 65K descriptors from the training dataset, 
that will be used for training the PCA Whitening  
##### arguments: 
* -m MODEL, --model MODEL
                        path of the model to load for performing inference; 
                        the PCA parameters will be saved to pca_[MODEL].h5
                        
*  -c CONFIGURATION, --configuration CONFIGURATION
                        Yaml file where the configuration is stored
*  -d DEVICE, --device DEVICE
                        CUDA device to be used. For info type `$ nvidia-smi`
                        
#### for extracting ROxford5K and RParis6K descriptors `$ python extract_features_revisitop.py`
* -m MODEL, --model MODEL
                        path of the model to load for performing inference; 
                        the PCA parameters will be saved to pca_[MODEL].h5
                        
*  -c CONFIGURATION, --configuration CONFIGURATION
                        Yaml file where the configuration is stored
*  -d DEVICE, --device DEVICE
                        CUDA device to be used. For info type `$ nvidia-smi`
*  -p for testing RParis6K (default option)
*  -o for testing ROxford5K

#### for computing mAPs of ROxf and RPar `$ python example_evaluate.py`
* -o for testing using ROxford5K descriptors
* -p for testing using RParis6K descriptors

#### Configuration file :
```
# architecture configuration
network:
  architecture: resnet101 # vgg16 or resnet101
  pooling_type: netvlad
  use_relu: true
  n_clusters: 64
  split_vlad: 512
  post_pca:
    active: false
  poolings:
    active: true
    pool_1_shape: 2
    pool_2_shape: 3
  middle_pca:
    active: false # don't use middle pca and pooling feature compression together!
    dim: 2048
    pretrain: false
  pooling_feature_compression:
    active: false
    pool_size: 6
    stride: 4
input-shape: 336

# mining parameters
threshold: 20
semi-hard-prob: 0.5
use_crop: false
mining_batch_size: 8000
images_per_class: 15

# training parameters
description: "resnet_relu_batch_size_8000_res512"
minibatch_size: 6
memory_saving: true
steps_per_epoch: 400
n_epochs: 400
checkpoint_freq: 20
lr:
  step_frequency: 120
  warm-up: true
  warm-up-steps: 2000 # if warm-up is active
  max_value: 1e-5
  min_value: 1e-6
  lr_decay: 2e-6

# testing (deprecated)
rotate_holidays: true
use_power_norm: false
use_multi_resolution: false
```
