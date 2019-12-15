import numpy as np
import vis.utils.utils
from keras import activations
from keras.applications import VGG16, ResNet50
from keras.layers import Input, Reshape, concatenate, MaxPool2D, Dense, Lambda
from keras.models import Model

from loupe_keras import NetVLAD
from triplet_loss import L2NormLayer

# from keras_vgg16_place.vgg16_places_365 import VGG16_Places365
# input_shape = (224, 224, 3)
input_shape = (336, 336, 3)
# input_shape = (504, 504, 3)


# vgg = VGG16(weights='imagenet', include_top=False, pooling=False, input_shape=input_shape)


class NetVLADSiameseModel:
    def __init__(self, layer_name='block5_conv2'):
        model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
        # model = VGG16_Places365(weights='places', include_top=False, pooling='avg', input_shape=input_shape)

        # set layers untrainable
        for layer in model.layers:
            layer.trainable = True
            # print(layer, layer.trainable)

        model.get_layer('block4_conv1').trainable = True
        model.get_layer('block4_conv2').trainable = True
        model.get_layer('block4_conv3').trainable = True

        model.get_layer('block5_conv1').trainable = True
        # model.get_layer('block5_conv2').trainable = True
        custom_layer = model.get_layer(layer_name)
        custom_layer.trainable = True

        model.get_layer(layer_name).activation = activations.linear
        model = vis.utils.utils.apply_modifications(model)

        # channel_red = Conv2D(256, 1)
        # out = channel_red(model.get_layer(layer_name).output)
        out = model.get_layer(layer_name).output

        n_filters = 512

        pool_1 = MaxPool2D(pool_size=2, strides=1, padding='valid')(out)
        pool_2 = MaxPool2D(pool_size=3, strides=1, padding='valid')(out)
        pool_3 = MaxPool2D(pool_size=4, strides=1, padding='valid')(out)

        out_reshaped = Reshape((-1, n_filters))(out)
        pool_1_reshaped = Reshape((-1, n_filters))(pool_1)
        pool_2_reshaped = Reshape((-1, n_filters))(pool_2)
        pool_3_reshaped = Reshape((-1, n_filters))(pool_3)

        out = concatenate([pool_1_reshaped, pool_2_reshaped], axis=1)
        # out = pool_1_reshaped

        self.backbone = model
        self.base_model = Model(model.input, out)

        self.layer_name = layer_name
        self.vgg_netvlad = None
        self.images_input = None
        self.filter_l = 14

    def get_feature_extractor(self, verbose=False):
        vgg = self.base_model
        vgg = Model(vgg.input, vgg.output)
        if verbose:
            vgg.summary()
        return vgg, vgg.output_shape

    def get_pooled_feature_extractor(self):
        self.images_input = Input(shape=input_shape)
        from keras.layers import AvgPool2D, Flatten

        filter_w = self.base_model.output_shape[1]
        filter_h = self.base_model.output_shape[2]

        pooled = AvgPool2D((filter_w, filter_h))(self.base_model([self.images_input]))
        flatten = Flatten()(pooled)

        return Model(inputs=self.images_input, output=flatten)

    def build_netvladmodel(self, kmeans=None):
        self.images_input = Input(shape=input_shape)

        self.anchor = Input(shape=input_shape)
        self.positive = Input(shape=input_shape)
        self.negative = Input(shape=input_shape)

        output_shape = self.base_model.output_shape
        n_filters = 512

        n_classes = 1

        # transpose = Permute((3, 1, 2), input_shape=(-1, n_filters))(self.base_model([self.images_input]))

        filter_l = self.filter_l

        # vgg_output = vgg.output_shape[1]
        # batch_norm = BatchNormalization()(self.base_model([self.images_input]))
        # reshape = Reshape((-1, n_filters))(batch_norm)
        l2normalization = L2NormLayer()(self.base_model.output)
        netvlad = NetVLAD(feature_size=n_filters, max_samples=filter_l ** 2, cluster_size=64)

        self.netvlad = netvlad

        netvlad = netvlad(l2normalization)

        netvlad_base = Model(self.base_model.input, netvlad)
        self.netvlad_base = netvlad_base
        # %%
        netvlad_a = netvlad_base([self.anchor])
        netvlad_p = netvlad_base([self.positive])
        netvlad_n = netvlad_base([self.negative])

        # embedding_output = netvlad(reshape(embedding(vgg.output)))
        siamese_output = concatenate(
            [netvlad_a, netvlad_p, netvlad_n]
        )

        vgg_netvlad = Model(inputs=[self.anchor, self.positive, self.negative],
                            outputs=[netvlad_a, netvlad_p, netvlad_n])

        if kmeans is not None:
            self.set_netvlad_weights(kmeans)

        self.vgg_netvlad = vgg_netvlad
        return self.vgg_netvlad

    def set_netvlad_weights(self, kmeans):
        # netvlad_ = self.vgg_netvlad.get_layer('net_vlad_1')
        netvlad_ = self.netvlad
        weights_netvlad = netvlad_.get_weights()
        # %%
        cluster_weights = kmeans.cluster_centers_
        alpha = 30.

        assignments_weights = 2. * alpha * cluster_weights
        assignments_bias = -alpha * np.sum(np.power(cluster_weights, 2), axis=1)

        cluster_weights = cluster_weights.T
        assignments_weights = assignments_weights.T
        assignments_bias = assignments_bias.T

        cluster_weights = np.expand_dims(cluster_weights, axis=0)
        # assignments_weights = np.expand_dims(assignments_weights, axis=0)
        # assignments_bias = np.expand_dims(assignments_bias, axis=0)
        # %%
        weights_netvlad[0] = assignments_weights
        weights_netvlad[1] = assignments_bias
        weights_netvlad[2] = cluster_weights

        netvlad_.set_weights(weights_netvlad)

    def get_netvlad_extractor(self):
        # return Model(inputs=self.images_input, outputs=self.netvlad)
        return self.netvlad_base


"""
from keras_retinanet.keras_retinanet import models
import os

class NetVLADModelRetinaNet(NetVLADModel):
    def __init__(self, layer_name='P4'):
        model_path = os.path.join('keras_retinanet', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

        # load retinanet model
        model = models.load_model(model_path, backbone_name='resnet50')

        # set layers untrainable
        for layer in model.layers:
            layer.trainable = False
            # print(layer, layer.trainable)
            #if layer.name is layer_name:
            #    layer.trainable = True

        custom_layer = model.get_layer(layer_name)
        custom_layer.trainable = True

        self.base_model = Model(model.input, custom_layer.output)

        self.vgg_netvlad = None
        self.images_input = None
        self.layer_name = layer_name

    """


# class NetVladResnet(NetVLADModel):
#     def __init__(self, layer_name='bn5c_branch2b'):
#         # model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
#         # model = VGG16_Places365(weights='places', include_top=False, pooling='avg', input_shape=input_shape)
#         model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#
#         # set layers untrainable
#         for layer in model.layers:
#             layer.trainable = False
#             # print(layer, layer.trainable)
#
#         # for layer in model.layers[-20:]:
#         #    layer.trainable = True
#
#         model.get_layer('res5b_branch2c').trainable = True
#         model.get_layer('bn5b_branch2c').trainable = True
#
#         model.get_layer('bn5c_branch2a').trainable = True
#         model.get_layer('res5c_branch2a').trainable = True
#
#         model.get_layer('res5c_branch2b').trainable = True
#         model.get_layer('bn5c_branch2b').trainable = True
#
#         # custom_layer = model.get_layer(layer_name)
#         # custom_layer.trainable = True
#
#         # model.get_layer(layer_name).activation = activations.linear
#         # model = vis.utils.utils.apply_modifications(model)
#
#         self.backbone = model
#         self.base_model = Model(model.input, model.get_layer(layer_name).output)
#
#         self.layer_name = layer_name
#         self.vgg_netvlad = None
#         self.images_input = None
#         self.filter_l = 7
#         self.netvlad_output = self.filter_l * self.filter_l * 64
