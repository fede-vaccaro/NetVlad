import numpy as np
import tensorflow as tf
import vis.utils.utils
from keras import activations
from keras import layers
from keras.applications import VGG16, ResNet50
from keras.layers import Input, Reshape, MaxPool2D
from keras.models import Model

from keras_gem import gem
from loupe_keras import NetVLAD
from triplet_loss import L2NormLayer


class NetVladBase:
    input_shape = (336, 336, 3)

    def __init__(self, **kwargs):
        self.output_layer = kwargs['output_layer']
        self.n_cluster = kwargs['n_clusters']
        self.middle_pca = kwargs['middle_pca']
        self.output_layer = kwargs['output_layer']

        self.poolings = kwargs['poolings']
        self.feature_compression = kwargs['pooling_feature_compression']

        self.regularizer = tf.keras.regularizers.l2(0.00001)

    def build_base_model(self, backbone):
        # backbone.summary()
        out = backbone.get_layer(self.output_layer).output
        print(out.shape)
        # self.n_filters = out.shape[3]
        self.n_filters = 512

        pool_1 = MaxPool2D(pool_size=self.poolings['pool_1_shape'], strides=1, padding='valid')(out)
        pool_2 = MaxPool2D(pool_size=self.poolings['pool_2_shape'], strides=1, padding='valid')(out)

        out_reshaped = Reshape((-1, self.n_filters))(out)
        pool_1_reshaped = Reshape((-1, self.n_filters))(pool_1)
        pool_2_reshaped = Reshape((-1, self.n_filters))(pool_2)

        if self.feature_compression['active']:
            feature_compression_pool_size = self.feature_compression['pool_size']
            feature_compression_strides = self.feature_compression['stride']

            max_pool_1d_1 = layers.AvgPool1D(pool_size=feature_compression_pool_size,
                                             strides=feature_compression_strides, data_format='channels_first')
            max_pool_1d_2 = layers.AvgPool1D(pool_size=feature_compression_pool_size,
                                             strides=feature_compression_strides, data_format='channels_first')

            pool_1_reshaped = max_pool_1d_1(pool_1_reshaped)
            pool_2_reshaped = max_pool_1d_2(pool_2_reshaped)

            self.n_filters = pool_2_reshaped.shape[2]

        out = layers.Concatenate(axis=1)([pool_1_reshaped, pool_2_reshaped])
        self.base_model = Model(backbone.input, out)
        self.siamese_model = None
        self.images_input = None
        self.filter_l = None  # useless, just for compatibility with netvlad implementation

    def get_feature_extractor(self, verbose=False):
        net = self.base_model
        net = Model(net.input, net.output)
        if verbose:
            net.summary()
        return net, net.output_shape

    def get_pooled_feature_extractor(self):
        self.images_input = Input(shape=self.input_shape)
        from keras.layers import AvgPool2D, Flatten

        filter_w = self.base_model.output_shape[1]
        filter_h = self.base_model.output_shape[2]

        pooled = AvgPool2D((filter_w, filter_h))(self.base_model([self.images_input]))
        flatten = Flatten()(pooled)

        return Model(inputs=self.images_input, output=flatten)

    def build_netvladmodel(self, kmeans=None):
        self.images_input = Input(shape=self.input_shape)

        feature_size = self.n_filters

        if self.middle_pca['active']:
            compression_dim = self.middle_pca['dim']
            pca = layers.Dense(compression_dim)
            self.pca = pca
            model_out = layers.Dropout(0.2)(self.base_model.output)
            pca = pca(model_out)
            l2normalization = L2NormLayer()(pca)

            feature_size = compression_dim
        else:
            l2normalization = L2NormLayer()(self.base_model.output)
            # l2normalization = self.base_model.output

        netvlad = NetVLAD(feature_size=feature_size, max_samples=0,
                          cluster_size=self.n_cluster)  # max samples is useless
        self.netvlad = netvlad
        netvlad = netvlad(l2normalization)

        netvlad_base = Model(self.base_model.input, netvlad)
        self.netvlad_base = netvlad_base

        if kmeans is not None:
            self.set_netvlad_weights(kmeans)

        self.siamese_model = self.get_siamese_network()
        return self.siamese_model

    def get_siamese_network(self):
        self.anchor = Input(shape=self.input_shape)
        self.positive = Input(shape=self.input_shape)
        self.negative = Input(shape=self.input_shape)

        netvlad_a = self.netvlad_base([self.anchor])
        netvlad_p = self.netvlad_base([self.positive])
        netvlad_n = self.netvlad_base([self.negative])
        siamese_model = Model(inputs=[self.anchor, self.positive, self.negative],
                              outputs=[netvlad_a, netvlad_p, netvlad_n])
        return siamese_model

    def set_mid_pca_weights(self, pca):
        if self.middle_pca['active']:
            mean_ = pca.mean_
            components_ = pca.components_

            mean_ = -np.dot(mean_, components_.T)
            self.pca.set_weights([components_.T, mean_])
        else:
            print("WARNING mid pca is not active")

    def set_netvlad_weights(self, kmeans):
        netvlad_ = self.netvlad
        weights_netvlad = netvlad_.get_weights()
        # %%
        cluster_weights = kmeans.cluster_centers_
        alpha = 30.0

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
        return self.netvlad_base


class NetVLADSiameseModel(NetVladBase):
    def __init__(self, **kwargs):
        model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=self.input_shape)
        super(NetVLADSiameseModel, self).__init__(**kwargs)

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
                        setattr(layer, attr, self.regularizer)
        else:
            # set layers untrainable
            for layer in model.layers:
                layer.trainable = True
                # print(layer, layer.trainable)
                for attr in ['kernel_regularizer']:
                    if hasattr(layer, attr):
                        setattr(layer, attr, self.regularizer)

        model.get_layer(self.output_layer).activation = activations.linear
        model = vis.utils.utils.apply_modifications(model)

        self.n_filters = None
        self.base_model = None
        self.siamese_model = None
        self.images_input = None
        self.filter_l = None  # useless, just for compatibility with netvlad implementation

        self.build_base_model(model)


class NetVladResnet(NetVladBase):
    def __init__(self, **kwargs):
        super(NetVladResnet, self).__init__(**kwargs)

        model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape, layers=tf.keras.layers)

        # set layers untrainable
        for layer in model.layers:
            if type(layer) is not type(layers.BatchNormalization()):
                layer.trainable = False

        self.regularizer = tf.keras.regularizers.l2(0.0001)

        for layer in model.layers:
            layer.trainable = True
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, self.regularizer)

        self.n_filters = None
        self.base_model = None
        self.siamese_model = None
        self.images_input = None
        self.filter_l = None  # useless, just for compatibility with netvlad implementation

        self.build_base_model(model)


class GeMResnet(NetVladResnet):
    def build_netvladmodel(self, kmeans=None):
        gem_out = gem.GeM(pool_size=11)(self.base_model.get_layer(self.output_layer).output)
        gem_out = layers.Flatten()(gem_out)
        gem_out = L2NormLayer()(gem_out)
        self.netvlad_base = Model(self.base_model.input, gem_out)

        self.siamese_model = self.get_siamese_network()
        return self.siamese_model

