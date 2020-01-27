import numpy as np
import tensorflow as tf
import vis.utils.utils
from keras import activations
from keras.applications import VGG16, ResNet50
from keras.layers import Input, Reshape, concatenate, MaxPool2D
from keras.models import Model

from loupe_keras import NetVLAD
from triplet_loss import L2NormLayer
import tensorflow as tf
from keras import layers

# from keras_vgg16_place.vgg16_places_365 import VGG16_Places365
# input_shape = (224, 224, 3)
# input_shape = (336, 336, 3)
input_shape = (None, None, 3)


# vgg = VGG16(weights='imagenet', include_top=False, pooling=False, input_shape=input_shape)


class NetVLADSiameseModel:
    def __init__(self, layer_name='block5_conv2'):
        model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
        # model = VGG16_Places365(weights='places', include_top=False, pooling='avg', input_shape=input_shape)

        regularizer = tf.keras.regularizers.l2(0.001)
        self.regularizer = regularizer
        few_layers = False

        if few_layers:
            for layer in model.layers:
                layer.trainable = False

            training_layers = [
                model.get_layer('block5_conv1'),
                model.get_layer('block5_conv2'),

                model.get_layer('block4_conv1'),
                model.get_layer('block4_conv2'),
                model.get_layer('block4_conv3'),
            ]

            # set layers untrainable
            for layer in training_layers:
                layer.trainable = True
                # print(layer, layer.trainable)
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)
        else:
            # set layers untrainable
            for layer in model.layers:
                layer.trainable = True
                # print(layer, layer.trainable)
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)


        model.get_layer(layer_name).activation = activations.linear
        model = vis.utils.utils.apply_modifications(model)

        # channel_red = Conv2D(256, 1)
        # out = channel_red(model.get_layer(layer_name).output)
        out = model.get_layer(layer_name).output

        self.n_filters = 512
        self.n_cluster = 64

        pool_1 = MaxPool2D(pool_size=2, strides=1, padding='valid')(out)
        pool_2 = MaxPool2D(pool_size=3, strides=1, padding='valid')(out)

        out_reshaped = Reshape((-1, self.n_filters))(out)
        pool_1_reshaped = Reshape((-1, self.n_filters))(pool_1)
        pool_2_reshaped = Reshape((-1, self.n_filters))(pool_2)

        out = concatenate([pool_1_reshaped, pool_2_reshaped], axis=1)

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

        filter_l = self.filter_l

        l2normalization = L2NormLayer()(self.base_model.output)
        netvlad = NetVLAD(feature_size=self.n_filters, max_samples=filter_l ** 2, cluster_size=self.n_cluster)

        self.netvlad = netvlad

        netvlad = netvlad(l2normalization)

        netvlad_base = Model(self.base_model.input, netvlad)
        self.netvlad_base = netvlad_base

        if kmeans is not None:
            self.set_netvlad_weights(kmeans)

        vgg_netvlad = self.get_siamese_network()

        self.vgg_netvlad = vgg_netvlad
        return self.vgg_netvlad

    def get_siamese_network(self):
        netvlad_a = self.netvlad_base([self.anchor])
        netvlad_p = self.netvlad_base([self.positive])
        netvlad_n = self.netvlad_base([self.negative])
        vgg_netvlad = Model(inputs=[self.anchor, self.positive, self.negative],
                            outputs=[netvlad_a, netvlad_p, netvlad_n])
        return vgg_netvlad

    def set_netvlad_weights(self, kmeans):
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

        weights_netvlad[0] = assignments_weights
        weights_netvlad[1] = assignments_bias
        weights_netvlad[2] = cluster_weights

        netvlad_.set_weights(weights_netvlad)

    def get_netvlad_extractor(self):
        # return Model(inputs=self.images_input, outputs=self.netvlad)
        return self.netvlad_base


# 'res4e_branch2a' x 80~
# res4a_branch2a x 77
# res4a_branch2b x 738
# res4b_branch2b x 829
# bn4b_branch2b x 845
# res4c_branch2a x 804
# bn4c_branch2a x 827
# res4c_branch2b
# bn4c_branch2b x 74.9
# bn5c_branch2a x 81.4
# bn5c_branch2a 32 cluster x 81.2
# bn5c_branch2a diretto x 78.3
# add_16 32 cluster x
# add_16 2048 -> 512 -> 84.7
# add_16 2048 -> 512 -> 64 cluter 85.0


class NetVladResnet(NetVLADSiameseModel):
    def __init__(self, layer_name='bn5c_branch2a'):
        model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, layers=tf.keras.layers)
        # set layers untrainable
        for layer in model.layers:
            if type(layer) is not type(layers.BatchNormalization()):
                layer.trainable = False

        regularizer = tf.keras.regularizers.l2(0.001)

        for layer in model.layers[-50:]:
            layer.trainable = True
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        model.summary()

        self.n_filters = 512
        self.n_cluster = 64

        out = model.get_layer(layer_name).output

        pool_1 = MaxPool2D(pool_size=2, strides=1, padding='valid')(out)
        pool_2 = MaxPool2D(pool_size=3, strides=1, padding='valid')(out)

        out_reshaped = Reshape((-1, self.n_filters))(out)
        pool_1_reshaped = Reshape((-1, self.n_filters))(pool_1)
        pool_2_reshaped = Reshape((-1, self.n_filters))(pool_2)

        out = concatenate([pool_1_reshaped, pool_2_reshaped], axis=1)

        self.backbone = model
        self.base_model = Model(model.input, out)

        # self.base_model.summary()

        self.layer_name = layer_name
        self.vgg_netvlad = None
        self.images_input = None
        self.filter_l = 7
        self.netvlad_output = self.n_filters * 32
