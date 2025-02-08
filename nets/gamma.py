from .block import BaseConv
from torch import Tensor
from torch import nn
import torch
def tanh_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return ((torch.tanh(x) * 0.5 + 0.5) * (right - left) + left)
        return activation
    return get_activation(l, r)
class Gamma(nn.Module):
    def __init__(self):
        super(Gamma, self).__init__()
        self.head1 = BaseConv(3, 16, ksize=3, stride=2)
        self.body1 = BaseConv(16, 16 * 2, ksize=3, stride=2)
        self.body3 = BaseConv(16 * 2, 16 , ksize=3, stride=2)
        self.image_adaptive_gamma = nn.Sequential(
            nn.Linear(16, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 3, bias=False)
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.gamma_range = [6., 10.5]
    def apply_gamma(self, img, params):
        params = tanh_range(self.gamma_range[0], self.gamma_range[1])(params)[..., None, None]
        out_image = img ** (1.0 / params)
        return Tensor(out_image)

    def forward(self, x):
        xin = torch.nn.functional.interpolate(x, (256, 256), mode='bilinear')
        fea = self.head1(xin)
        fea_s2 = self.body1(fea)
        fea_s8 = self.body3(fea_s2)
        fea_gamma = self.pooling(fea_s8)
        fea_gamma = fea_gamma.view(fea_gamma.shape[0], fea_gamma.shape[1])
        para_gamma = self.image_adaptive_gamma(fea_gamma)
        out_gamma = self.apply_gamma(x, para_gamma)
        out = torch.clamp(out_gamma,0,1)
        return out
