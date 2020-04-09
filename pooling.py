import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

# --------------------------------------
# Pooling layers
# --------------------------------------
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class GeMmp(nn.Module):

    def __init__(self, p=3, mp=1, eps=1e-6):
        super(GeMmp,self).__init__()
        self.p = Parameter(torch.ones(mp)*p)
        self.mp = mp
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '[{}]'.format(self.mp) + ', ' + 'eps=' + str(self.eps) + ')'
