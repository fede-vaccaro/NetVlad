import numpy as np
import vis.utils.utils
from keras import activations
from keras.applications import VGG16
from keras.layers import Input, Reshape, concatenate, Permute
from keras.models import Model

from loupe_keras import NetVLAD
from triplet_loss import L2NormLayer

input_shape = (224, 224, 3)

# vgg = VGG16(weights='imagenet', include_top=False, pooling=False, input_shape=input_shape)
vgg = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)


class NetVLADModel:

    def __init__(self, layer_name='block5_conv2'):
        vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)

        # set layers untrainable
        for layer in vgg_model.layers:
            layer.trainable = False
            # print(layer, layer.trainable)
            if layer.name is layer_name:
                layer.trainable = True

        vgg_model.get_layer(layer_name).activation = activations.linear
        vgg_model = vis.utils.utils.apply_modifications(vgg_model)

        self.base_model = vgg = Model(vgg_model.input, vgg_model.get_layer('block5_conv2').output)
        self.vgg_netvlad = None
        self.images_input = None

    def get_feature_extractor(self, verbose=False):
        vgg = self.base_model
        vgg = Model(vgg.input, vgg.get_layer('block5_conv2').output)
        if verbose:
            vgg.summary()
        return vgg, vgg.output_shape

    def build_netvladmodel(self, n_classes, kmeans):
        self.images_input = Input(shape=(224, 224, 3))
        label_input = Input(shape=(n_classes,), name="input_label")

        transpose = Permute((3, 1, 2), input_shape=(-1, 512))(self.base_model([self.images_input]))
        embedding_size = 512

        # vgg_output = vgg.output_shape[1]
        # embedding = Dense(embedding_size, input_shape=(vgg_output,), activation='relu', name="embedding1")(vgg([images_input]))
        reshape = Reshape((512, 14 * 14))(transpose)
        l2normalization = L2NormLayer()(reshape)
        netvlad = NetVLAD(feature_size=14 * 14, max_samples=512, cluster_size=64)(
            l2normalization)  # , output_dim=1024)resnet_output = resnet.output_shape[1]

        netvlad_output = 14 * 14 * 64

        # %%

        # embedding_output = netvlad(reshape(embedding(vgg.output)))
        labels_plus_embeddings = concatenate([label_input, netvlad])

        vgg_netvlad = Model(inputs=[self.images_input, label_input], outputs=labels_plus_embeddings)

        # %%
        netvlad_ = vgg_netvlad.get_layer('net_vlad_1')
        weights_netvlad = netvlad_.get_weights()

        # %%
        cluster_weights = kmeans.cluster_centers_
        alpha = 20.
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

        self.vgg_netvlad = vgg_netvlad
        return self.vgg_netvlad, netvlad_output

    def get_netvlad_extractor(self):
        return Model(inputs=self.images_input, outputs=self.vgg_netvlad.get_layer('net_vlad_1').output)
