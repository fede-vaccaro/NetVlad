import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).norm(p=2, dim=1)  # .pow(.5)
        distance_negative = (anchor - negative).norm(p=2, dim=1)  # .pow(.5)
        losses = distance_positive.pow(2.0) + torch.exp(-distance_negative)
        return losses.mean()
