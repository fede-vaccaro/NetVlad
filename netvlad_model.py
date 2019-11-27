import numpy as np
import vis.utils.utils
from keras import activations
from keras.applications import VGG16
from keras.layers import Input, Reshape, concatenate, Permute
from keras.models import Model

from loupe_keras import NetVLAD
from triplet_loss import L2NormLayer
from keras_vgg16_place.vgg16_places_365 import VGG16_Places365
input_shape = (224, 224, 3)

# vgg = VGG16(weights='imagenet', include_top=False, pooling=False, input_shape=input_shape)


class NetVLADModel:
    def __init__(self, layer_name='block5_conv3'):
        # model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
        model = VGG16_Places365(weights='places', include_top=False, pooling='avg', input_shape=input_shape)

        # set layers untrainable
        for layer in model.layers:
            layer.trainable = False
            # print(layer, layer.trainable)

        custom_layer = model.get_layer(layer_name)
        custom_layer.trainable = True

        model.get_layer(layer_name).activation = activations.linear
        model = vis.utils.utils.apply_modifications(model)

        self.backbone = model
        self.base_model = Model(model.input, model.get_layer(layer_name).output)

        self.layer_name = layer_name
        self.vgg_netvlad = None
        self.images_input = None
        self.filter_l = 14
        self.netvlad_output = self.filter_l*self.filter_l*128

    def get_feature_extractor(self, verbose=False):
        vgg = self.base_model
        vgg = Model(vgg.input, vgg.get_layer(self.layer_name).output)
        if verbose:
            vgg.summary()
        return vgg, vgg.output_shape

    def get_pooled_feature_extractor(self):
        self.images_input = Input(shape=(224, 224, 3))
        from keras.layers import AvgPool2D, Flatten

        filter_w = self.base_model.output_shape[1]
        filter_h = self.base_model.output_shape[2]


        pooled = AvgPool2D((filter_w, filter_h))(self.base_model([self.images_input]))
        flatten = Flatten()(pooled)

        return Model(inputs=self.images_input, output=flatten)

    def build_netvladmodel(self, n_classes, kmeans=None):
        self.images_input = Input(shape=(224, 224, 3))
        output_shape = self.base_model.output_shape
        n_filters = output_shape[3]

        label_input = Input(shape=(n_classes,), name="input_label")

        transpose = Permute((3, 1, 2), input_shape=(-1, n_filters))(self.base_model([self.images_input]))

        filter_l = self.filter_l

        # vgg_output = vgg.output_shape[1]
        reshape = Reshape((n_filters, filter_l * filter_l))(transpose)
        l2normalization = L2NormLayer()(reshape)
        netvlad = NetVLAD(feature_size=filter_l * filter_l, max_samples=n_filters, cluster_size=128)(
            l2normalization)  # , output_dim=1024)resnet_output = resnet.output_shape[1]


        # %%

        # embedding_output = netvlad(reshape(embedding(vgg.output)))
        labels_plus_embeddings = concatenate([label_input, netvlad])

        vgg_netvlad = Model(inputs=[self.images_input, label_input], outputs=labels_plus_embeddings)

        if kmeans is not None:
            self.set_netvlad_weights(kmeans)

        self.vgg_netvlad = vgg_netvlad
        return self.vgg_netvlad

    def set_netvlad_weights(self, kmeans):
        netvlad_ = self.vgg_netvlad.get_layer('net_vlad_1')
        weights_netvlad = netvlad_.get_weights()
        # %%
        cluster_weights = kmeans.cluster_centers_
        alpha = 10.

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
        return Model(inputs=self.images_input, outputs=self.vgg_netvlad.get_layer('net_vlad_1').output)


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

    """ def get_feature_extractor(self, verbose=False):
        vgg = self.base_model
        #vgg = Model(vgg.input, vgg.get_layer('block5_conv2').output)
        if verbose:
            vgg.summary()
        return vgg, vgg.output_shape

    def build_netvladmodel(self, n_classes, kmeans):
        pass
    def get_netvlad_extractor(self):
        return self.get_feature_extractor()
        #return Model(inputs=self.images_input, outputs=self.vgg_netvlad.get_layer('net_vlad_1').output)
    """

