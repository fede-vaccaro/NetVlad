import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from netvlad import NetVLAD, make_locals
from pooling import GeM


def normalize_torch(x):
    return torch.nn.functional.normalize(x, dim=2, p=2)


class NetVladBase(nn.Module):
    input_shape = (336, 336, 3)

    def __init__(self, **kwargs):
        super(NetVladBase, self).__init__()
        self.n_cluster = kwargs['n_clusters']
        self.middle_pca = kwargs['middle_pca']
        self.split_vlad = kwargs['split_vlad']
        self.pooling_type = kwargs['pooling_type']
        self.poolings = kwargs['poolings']
        self.feature_compression = kwargs['pooling_feature_compression']

        if self.middle_pca['active']:
            self.conv_1 = torch.nn.Conv2d(2048, self.middle_pca['dim'], kernel_size=1, stride=1)
            self.conv_1.cuda()

        self.n_filters = None

        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=self.input_shape[:2], interpolation=Image.ANTIALIAS),
            torchvision.transforms.ToTensor(),
        ])
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(504, 504), interpolation=Image.ANTIALIAS),
            torchvision.transforms.RandomCrop(size=(336, 336)),
            torchvision.transforms.ToTensor(),
            normalize
        ])
        full_transform = torchvision.transforms.Compose([
            transform,
            normalize,
        ])
        self.normalize = normalize
        self.transform = transform
        self.full_transform = full_transform
        self.train_transform = train_transform

    def get_transform(self, shape):
        if type(shape) is type(1):
            shape_ = (shape, shape)
        else:
            shape_ = shape
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=shape_, interpolation=Image.ANTIALIAS),
            torchvision.transforms.ToTensor(),
        ])
        full_transform = torchvision.transforms.Compose([
            transform,
            normalize,
        ])
        return full_transform

    def init_pooling_layer(self):
        if self.pooling_type == 'netvlad':
            self.netvlad_pool = NetVLAD(num_clusters=self.n_cluster, dim=self.n_filters)
            self.output_dim = self.netvlad_pool.output_dim
        elif self.pooling_type == 'gem':
            self.netvlad_pool = GeM()
            self.output_dim = 2048

    def predict_with_netvlad(self, img_tensor, batch_size=16, verbose=False):

        n_imgs = img_tensor.shape[0]
        descs = np.zeros((n_imgs, self.output_dim))
        n_iters = int(np.ceil(n_imgs / batch_size))

        with torch.no_grad():
            self.eval()
            if verbose:
                print("")
            for i in range(n_iters):
                low = i * batch_size
                high = int(np.min([n_imgs, (i + 1) * batch_size]))
                batch_gpu = img_tensor[low:high].cuda()
                out_batch = self.forward(batch_gpu).cpu().numpy()
                descs[low:high] = out_batch
                if verbose:
                    print("\r>> Predicted batch {}/{}".format(i + 1, n_iters), end='')
            if verbose:
                print("")

        return descs

    def predict_generator_with_netlvad(self, generator, n_steps, verbose=True):

        descs = []

        t0 = time.time()
        with torch.no_grad():
            self.eval()
            print("")
            for i, X in enumerate(generator):
                if type(X) is type(tuple()) or type(X) is type(list()):
                    x = X[0]
                else:
                    x = X
                batch_gpu = x.cuda()
                out_batch = self.forward(batch_gpu).cpu().numpy()
                descs.append(out_batch)
                if verbose:
                    print("\r>> Predicted (w/ generator) batch {}/{}".format(i + 1, n_steps), end='')
                if i + 1 == n_steps:
                    break
            if verbose:
                print("\n>> Prediction completed in {}s".format(int(time.time() - t0)))

        descs = np.vstack([m for m in descs])
        return descs

    def features(self, x):
        # backbone.summary()
        x = self.base_features(x)
        if self.middle_pca['active']:
            x = self.conv_1(x)

        pool_1 = nn.functional.max_pool2d(x, kernel_size=self.poolings['pool_1_shape'], stride=1)
        pool_2 = nn.functional.max_pool2d(x, kernel_size=self.poolings['pool_2_shape'], stride=1)

        # pool_1 = nn.functional.adaptive_max_pool2d(x, output_size=(10,10))
        # pool_2 = nn.functional.adaptive_max_pool2d(x, output_size=(9,9))

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
        if self.pooling_type == 'gem':
            if self.poolings['active']:
                p1, p2 = self.features(x)
                x1 = self.netvlad_pool(p1).squeeze(-1).squeeze(-1)
                x2 = self.netvlad_pool(p2).squeeze(-1).squeeze(-1)
                cat = torch.cat([x1, x2], dim=1)
                out = torch.nn.functional.normalize(cat, dim=1, p=2)
            else:
                x = self.base_features(x)
                x = self.netvlad_pool(x).squeeze(-1).squeeze(-1)
                out = torch.nn.functional.normalize(x, dim=1, p=2)
        elif self.pooling_type == 'netvlad':
            x = self.features(x)
            out = self.netvlad_pool(x)
        else:
            raise ValueError('Pooling not recognized: {}'.format(self.pooling_type))

        return out

    def get_siamese_output(self, a, p, n):
        d_a = self.forward(a)
        d_p = self.forward(p)
        d_n = self.forward(n)

        return d_a, d_p, d_n

    def set_whitening_weights(self, pca):
        if self.middle_pca['active']:
            mean_ = pca.mean_
            components_ = pca.components_
            print(components_.shape)
            mean_ = -np.dot(mean_, components_.T)
            components_ = components_.reshape(self.middle_pca['dim'], 2048, 1, 1)

            self.conv_1.weight = torch.nn.Parameter(torch.Tensor(components_), requires_grad=True)
            self.conv_1.bias = torch.nn.Parameter(torch.Tensor(mean_), requires_grad=True)
            self.conv_1.cuda()
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

    def initialize_whitening(self, image_folder):
        print("Predicting local features for mid whitening.")
        # all_descs = self.get_feature_extractor(verbose=True)[0].predict_generator(generator=kmeans_generator, steps=30,
        #                                                                           verbose=1)

        n_batches = 30
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
                desc = self.base_features(x.cuda())
                # N, dim, h, w = desc.shape
                # desc = desc.view(N, dim, h*w).permute(0, 2, 1).reshape(N, -1, 512)
                desc = desc.cpu().numpy().astype('float32').reshape(-1, 2048)
                descs_list.append(desc)
                print("\r>> Extracted batch {}/{} - NetVLAD initialization -".format(i + 1, n_batches), end='')
                i += 1
                if i == n_batches:
                    break
            print("")

        # locals = np.vstack((m[np.random.randint(len(m), size=150)] for m in descs_list)).astype('float32')
        descs_list = np.array(descs_list)
        desc_dim = descs_list.shape[-1]
        locals = descs_list.reshape(-1, desc_dim)
        n_locals, dim = locals.shape
        # locals = locals[np.random.randint(n_locals, size=n_locals//3)]
        print("{} Local activations of dim {}".format(n_locals, dim))

        np.random.shuffle(locals)

        print("Training Scikit-Learn PCA @ dim: ", self.middle_pca['dim'])
        pca = PCA(self.middle_pca['dim'])
        pca.fit(locals)

        self.set_whitening_weights(pca)
        print("Whitening layer initialized")

        del descs_list
        gc.collect()

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
            print("")

        # locals = np.vstack((m[np.random.randint(len(m), size=150)] for m in descs_list)).astype('float32')
        descs_list = np.array(descs_list)
        desc_dim = descs_list.shape[-1]
        locals = descs_list.reshape(-1, desc_dim)
        n_locals, dim = locals.shape
        # locals = locals[np.random.randint(n_locals, size=n_locals//3)]
        print("{} Local features of dim {}".format(n_locals, dim))

        np.random.shuffle(locals)

        if False:
            # if self.middle_pca['pretrain'] and self.middle_pca['active']:
            print("Training PCA")
            pca = PCA(self.middle_pca['dim'])
            locals = pca.fit_transform(locals)
            self.set_whitening_weights(pca)

        print("Locals extracted: {}".format(locals.shape))

        n_clust = self.n_cluster
        # locals = normalize(locals, axis=1)

        print("Fitting k-means")
        kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=424242).fit(locals)

        self.set_netvlad_weights(kmeans)
        print("NetVLAD layer initialized")

        del descs_list
        gc.collect()


class VLADNet(NetVladBase):
    def __init__(self, **kwargs):
        super(VLADNet, self).__init__(**kwargs)

        arch_name = kwargs['architecture']
        if arch_name == 'resnet101':
            model = torchvision.models.resnet101(pretrained=False)

            if not kwargs['use_relu']:
                model.layer4[2].relu = nn.Identity()

            # courtesy of F. Radenoviç http://cmp.felk.cvut.cz/cnnimageretrieval/
            file_model = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth'

            base_features = list(model.children())[:-2]

        elif arch_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=False)
            base_features = list(model.features.children())[:-1]
            self.netvlad_out_dim = 512
            file_model = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth'
        else:
            raise ValueError('Architecture "{}" not available.'.format(arch_name))

        # sobstitute relu with Identity

        # get base_features
        base_features = nn.Sequential(*base_features)
        base_features.load_state_dict(model_zoo.load_url(file_model, model_dir=os.getcwd()))
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
        self.n_filters = self.split_vlad

        self.init_pooling_layer()
        # self.build_base_model(model)
