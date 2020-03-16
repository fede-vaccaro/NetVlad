import gc

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from netvlad import NetVLAD, make_locals


def normalize_torch(x):
    return torch.nn.functional.normalize(x, dim=2, p=2)


class NetVladBase(nn.Module):
    input_shape = (336, 336, 3)

    def __init__(self, **kwargs):
        super(NetVladBase, self).__init__()
        self.output_layer = kwargs['output_layer']
        self.n_cluster = kwargs['n_clusters']
        self.middle_pca = kwargs['middle_pca']
        self.output_layer = kwargs['output_layer']
        self.n_splits = kwargs['split_vlad']

        self.poolings = kwargs['poolings']
        self.feature_compression = kwargs['pooling_feature_compression']

        # self.regularizer = tf.keras.regularizers.l2(0.001)

        self.n_filters = None

        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(336, 336)),
            torchvision.transforms.ToTensor(),
        ])
        full_transform = torchvision.transforms.Compose([
            transform,
            normalize,
        ])
        self.normalize = normalize
        self.transform = transform
        self.full_transform = full_transform

    def init_vlad(self):
        self.netvlad_pool = NetVLAD(num_clusters=self.n_cluster, dim=self.n_filters)
        self.netvlad_out_dim = self.netvlad_pool.output_dim

    def predict_with_netvlad(self, img_tensor, batch_size=16):

        n_imgs = img_tensor.shape[0]
        descs = np.zeros((n_imgs, self.netvlad_out_dim))
        n_iters = int(np.ceil(n_imgs/batch_size))

        with torch.no_grad():
            self.eval()
            for i in range(n_iters):
                low = i*batch_size
                high = int(np.min([n_imgs, (i + 1) * batch_size]))
                batch_gpu = img_tensor[low:high].cuda()
                out_batch = self.forward(batch_gpu).cpu().numpy()
                descs[low:high] = out_batch
                print("\rPredicted batch {}/{}".format(i+1, n_iters), end='')

        return descs



    def features(self, x):
        # backbone.summary()
        x = self.base_features(x)

        pool_1 = nn.functional.max_pool2d(x, kernel_size=3, stride=1)
        pool_2 = nn.functional.max_pool2d(x, kernel_size=2, stride=1)

        out_reshaped = make_locals(x, n_filters=self.n_filters)
        pool_1_reshaped = make_locals(pool_1, n_filters=self.n_filters)
        pool_2_reshaped = make_locals(pool_2, n_filters=self.n_filters)

        if self.feature_compression['active']:
            pass

        if self.poolings['active']:
            out = torch.cat([pool_1_reshaped, pool_2_reshaped], dim=1)
        else:
            out = out_reshaped

        out = normalize_torch(out)

        # for i in range(self.n_splits):
        #     split_i = layers.Lambda(lambda x: x[:, :, i * self.split_dimension:(i + 1) * self.split_dimension])(out)
        #     self.out_splits.append(split_i)
        #     print("Out shape: {}, split shape: {}".format(out.shape, split_i.shape))

        return out

        # self.base_model = Model(backbone.input, [split for split in self.out_splits])
        # self.siamese_model = None
        # self.images_input = None
        # self.filter_l = None  # useless, just for compatibility with netvlad implementation

    def avg_pooled_features(self, x):
        x = self.base_features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        return out

    def forward(self, x):
        x = self.features(x)
        out = self.netvlad_pool(x)

        return out

    # def build_netvladmodel(self, kmeans=None):
    #     netvlad_out = []
    #     self.netvlad = []
    #
    #     for split in self.out_splits:
    #         feature_size = self.n_filters
    #
    #         if self.middle_pca['active']:
    #             compression_dim = self.middle_pca['dim']
    #             pca = layers.Dense(compression_dim)
    #             self.pca = pca
    #             model_out = layers.Dropout(0.2)(self.base_model.output)
    #             pca = pca(model_out)
    #             l2normalization = L2NormLayer()(pca)
    #
    #             feature_size = compression_dim
    #         else:
    #             # l2normalization = L2NormLayer()(split)
    #             l2normalization = split
    #             # l2normalization = self.base_model.output
    #
    #         netvlad = NetVLAD(feature_size=self.split_dimension, max_samples=0,
    #                           cluster_size=self.n_cluster)  # max samples is useless
    #
    #         self.netvlad += [netvlad]
    #
    #         netvlad_i = netvlad(l2normalization)
    #
    #         netvlad_out.append(netvlad_i)
    #
    #     if len(netvlad_out) > 1:
    #         netvlad_base = Model(self.base_model.input,
    #                              L2NormLayer()(concatenate([netvlad for netvlad in netvlad_out])))
    #     else:
    #         netvlad_base = Model(self.base_model.input, L2NormLayer()(netvlad_out[0]))
    #         # netvlad_base = Model(self.base_model.input, netvlad_out[0])
    #     self.netvlad_base = netvlad_base
    #
    #     # self.netvlad_base.summary()
    #     if kmeans is not None:
    #         self.set_netvlad_weights(kmeans)
    #
    #     self.siamese_model = self.get_siamese_network()
    #     return self.siamese_model

    def get_siamese_output(self, a, p, n):
        # self.images_input = Input(shape=self.input_shape)
        #
        # self.anchor = Input(shape=self.input_shape)
        # self.positive = Input(shape=self.input_shape)
        # self.negative = Input(shape=self.input_shape)
        #
        # netvlad_a = self.netvlad_base([self.anchor])
        # netvlad_p = self.netvlad_base([self.positive])
        # netvlad_n = self.netvlad_base([self.negative])
        # siamese_model = Model(inputs=[self.anchor, self.positive, self.negative],
        #                       outputs=[netvlad_a, netvlad_p, netvlad_n])

        d_a = self.forward(a)
        d_p = self.forward(p)
        d_n = self.forward(n)

        return d_a, d_p, d_n

    def set_mid_pca_weights(self, pca):
        if self.middle_pca['active']:
            mean_ = pca.mean_
            components_ = pca.components_

            mean_ = -np.dot(mean_, components_.T)
            self.pca.set_weights([components_.T, mean_])
        else:
            print("WARNING mid pca is not active")

    def set_netvlad_weights(self, kmeans):
        # netvlad_ = self.netvlad[split_index]
        # weights_netvlad = netvlad_.get_weights()
        # %%
        centroids = kmeans.cluster_centers_
        self.netvlad_pool.init_params(centroids.T)

        # alpha = self.netvlad.alpha
        #
        # assignments_weights = 2. * alpha * centroids
        # assignments_bias = -alpha * np.sum(np.power(centroids, 2), axis=1)
        #
        # centroids = centroids.T
        # assignments_weights = assignments_weights.T
        # assignments_bias = assignments_bias.T
        #
        # centroids = np.expand_dims(centroids, axis=0)
        # assignments_weights = np.expand_dims(assignments_weights, axis=0)
        # assignments_bias = np.expand_dims(assignments_bias, axis=0)

        # weights_netvlad[0] = assignments_weights
        # weights_netvlad[1] = assignments_bias
        # weights_netvlad[2] = centroids
        #
        # netvlad_.set_weights(weights_netvlad)
        self.netvlad_pool.cuda()

    # def get_netvlad_extractor(self):
    #     return self.netvlad_base

    def initialize_netvlad(self, image_folder):
        print("Predicting local features for k-means.")
        # all_descs = self.get_feature_extractor(verbose=True)[0].predict_generator(generator=kmeans_generator, steps=30,
        #                                                                           verbose=1)

        n_batches = 10
        train_loader = torch.utils.data.DataLoader(
            image_folder,
            batch_size=32,
            num_workers=8,
            shuffle=True,
        )

        descs_list = []
        i = 0
        with torch.no_grad():
            self.eval()
            for x, _ in train_loader:
                desc = self.features(x.cuda())
                # N, dim, h, w = desc.shape
                # desc = desc.view(N, dim, h*w).permute(0, 2, 1).reshape(N, -1, 512)
                desc = desc.cpu().numpy().astype('float32')
                descs_list.append(desc)
                print("\r>> Extracted batch {}/{} - NetVLAD initialization -".format(i + 1, n_batches), end='')
                i += 1
                if i == n_batches:
                    break

        # locals = np.vstack((m[np.random.randint(len(m), size=150)] for m in descs_list)).astype('float32')
        descs_list = np.array(descs_list)
        desc_dim = descs_list.shape[-1]
        locals = descs_list.reshape(-1, desc_dim)
        n_locals, dim = locals.shape
        locals = locals[np.random.randint(n_locals, size=n_locals//3)]
        print("{} Local features of dim {}".format(n_locals, dim))

        np.random.shuffle(locals)

        if self.middle_pca['pretrain'] and self.middle_pca['active']:
            print("Training PCA")
            pca = PCA(self.middle_pca['dim'])
            locals = pca.fit_transform(locals)
            self.set_mid_pca_weights(pca)

        print("Locals extracted: {}".format(locals.shape))

        n_clust = self.n_cluster
        locals = normalize(locals, axis=1)

        print("Fitting k-means")
        kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=424242).fit(locals)

        print("Initializing NetVLAD")
        self.set_netvlad_weights(kmeans)

        del descs_list
        gc.collect()


# class NetVLADSiameseModel(NetVladBase):
#     def __init__(self, **kwargs):
#         model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=self.input_shape)
#         super(NetVLADSiameseModel, self).__init__(**kwargs)
#
#         few_layers = False
#         if few_layers:
#             for layer in model.layers:
#                 layer.trainable = False
#
#             training_layers = [
#                 model.get_layer('block5_conv1'),
#                 model.get_layer('block5_conv2'),
#
#                 model.get_layer('block4_conv1'),
#                 model.get_layer('block4_conv2'),
#                 model.get_layer('block4_conv3'),
#             ]
#
#             # set layers untrainable
#             for layer in training_layers:
#                 layer.trainable = True
#                 # print(layer, layer.trainable)
#                 for attr in ['kernel_regularizer']:
#                     if hasattr(layer, attr):
#                         setattr(layer, attr, self.regularizer)
#         else:
#             # set layers untrainable
#             for layer in model.layers:
#                 layer.trainable = True
#                 # print(layer, layer.trainable)
#                 for attr in ['kernel_regularizer']:
#                     if hasattr(layer, attr):
#                         setattr(layer, attr, self.regularizer)
#
#         model.get_layer(self.output_layer).activation = activations.linear
#         model = vis.utils.utils.apply_modifications(model)
#
#         self.n_filters = None
#         self.base_model = None
#         self.siamese_model = None
#         self.images_input = None
#         self.filter_l = None  # useless, just for compatibility with netvlad implementation
#
#         self.build_base_model(model)


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


class NetVladResnet(NetVladBase):
    def __init__(self, **kwargs):
        super(NetVladResnet, self).__init__(**kwargs)

        model = getattr(torchvision.models, 'resnet101')(pretrained=False)

        # sobstitute relu with Identity
        model.layer4[2].relu = nn.Identity()

        # get base_features
        base_features = list(model.children())[:-2]
        base_features = nn.Sequential(*base_features)
        base_features.load_state_dict(torch.load("imagenet-caffe-resnet101-features-10a101d.pth"))

        # self.regularizer = tf.keras.regularizers.l2(0.001)
        #
        # for layer in model.layers:
        #     layer.trainable = True
        #     for attr in ['kernel_regularizer']:
        #         if hasattr(layer, attr):
        #             setattr(layer, attr, self.regularizer)

        self.base_features = base_features.cuda()
        self.base_model = None
        self.siamese_model = None
        self.images_input = None
        self.n_filters = 512

        self.init_vlad()
        # self.build_base_model(model)
