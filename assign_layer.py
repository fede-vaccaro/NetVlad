from keras import backend as K
from keras import layers, initializers, regularizers


class AssignLayer(layers.Layer):
    def __init__(self, softness, n_clusters, use_bias=True,
                 use_normalization=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        self.softness = softness
        self.n_clusters = n_clusters
        self.use_bias = use_bias
        self.use_normalization = use_normalization

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)

        super(AssignLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.centroids = self.add_weight(shape=(input_dim, self.n_clusters),
                                         initializer=self.kernel_initializer,
                                         name='kernel',
                                         regularizer=self.kernel_regularizer)
        # constraint=self.kernel_constraint)
        super(AssignLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        if self.use_normalization:
            output = K.dot(inputs, 2.0 * self.softness * K.l2_normalize(self.centroids, axis=0))
        else:
            output = K.dot(inputs, 2.0 * self.softness * self.centroids)
        if self.use_bias:
            bias = - self.softness * K.sum(K.pow(self.centroids, 2), axis=0)
            output = K.bias_add(output, bias, data_format='channels_last')

        return output
