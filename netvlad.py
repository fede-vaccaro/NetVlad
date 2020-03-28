import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from https://github.com/lyakaap/NetVLAD-pytorch

# class NetVLAD(nn.Module):
#     """NetVLAD layer implementation"""
#
#     def __init__(self, num_clusters=16, dim=2048, alpha=30.0,
#                  normalize_input=True):
#         """
#         Args:
#             num_clusters : int
#                 The number of clusters
#             dim : int
#                 Dimension of descriptors
#             alpha : float
#                 Parameter of initialization. Larger value is harder assignment.
#             normalize_input : bool
#                 If true, descriptor-wise L2 normalization is applied to input.
#         """
#         super(NetVLAD, self).__init__()
#         self.num_clusters = num_clusters
#         self.dim = dim
#         self.alpha = alpha
#         self.normalize_input = normalize_input
#         self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
#         self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
#         self._init_params()
#
#     def _init_params(self):
#         self.conv.weight = nn.Parameter(
#             (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
#         )
#         self.conv.bias = nn.Parameter(
#             - self.alpha * self.centroids.norm(dim=1)
#         )
#
#     def forward(self, x):
#         N, C = x.shape[:2]
#         if self.normalize_input:
#             x = F.normalize(x, p=2, dim=1)  # across descriptor dim
#
#         # soft-assignment
#         soft_assign = self.conv(x).view(N, self.num_clusters, -1)
#         soft_assign = F.softmax(soft_assign, dim=1)
#
#         x_flatten = x.view(N, C, -1)
#
#         # calculate residuals to each clusters
#         residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
#                    self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
#         residual *= soft_assign.unsqueeze(2)
#         vlad = residual.sum(dim=-1)
#
#         vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
#         vlad = vlad.view(x.size(0), -1)  # flatten
#         vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
#
#         return vlad


def make_locals(x, n_filters=512):
    N, C, W, H = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(N, -1, n_filters)
    return x


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=512, alpha=30.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input

        self.centroids = nn.Parameter(torch.rand(dim, num_clusters), requires_grad=True)
        self.assignment_weights = nn.Parameter(
            (2.0 * self.alpha * self.centroids)  # .unsqueeze(-1).unsqueeze(-1)
            , requires_grad=True)
        self.assignment_bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=0)
            , requires_grad=True)
        self.output_dim = num_clusters*dim

    def get_output_dim(self):
        return self.output_dim

    def init_params(self, centroids: np.array):
        self.centroids = nn.Parameter(torch.Tensor(centroids), requires_grad=True)
        self.assignment_weights = nn.Parameter(
            (2.0 * self.alpha * self.centroids)  # .unsqueeze(-1).unsqueeze(-1)
            , requires_grad=True)
        self.assignment_bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=0)
            , requires_grad=True)

    def forward(self, x):
        # assignment weights = D x K
        # centroids = D x K

        # max_pooled_feat_3 = F.max_pool2d(x, kernel_size=3, stride=1)
        # max_pooled_feat_2 = F.max_pool2d(x, kernel_size=2, stride=1)
        #
        # reshaped_pool_3 = make_locals(max_pooled_feat_3)
        # reshaped_pool_2 = make_locals(max_pooled_feat_2)
        #
        # x = torch.cat([reshaped_pool_2, reshaped_pool_3], dim=1)

        # x = make_locals(x)

        # if self.normalize_input:
        #     x = F.normalize(x, p=2, dim=2)  # across descriptor dim

        # soft-assignment
        soft_assign = torch.matmul(x, self.assignment_weights) + self.assignment_bias
        soft_assign = F.softmax(soft_assign, dim=2)

        # soft_assign dim:

        # calculate residuals to each clusters

        a_sum = torch.sum(soft_assign, dim=1).unsqueeze(1)
        a = a_sum * self.centroids
        soft_assign = soft_assign.permute((0, 2, 1))

        vlad = torch.matmul(soft_assign, x)
        vlad = vlad.permute((0, 2, 1))
        vlad = vlad - a

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

    def __repr__(self):
        return self.__class__.__name__ + " k={}, dim={}".format(self.num_clusters, self.dim)


class EmbedNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        return embedded_x


class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)
