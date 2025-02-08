from torch import Tensor
import numpy as np
import torch
class GrayWorldWB:
    def __init__(self):
        self.clip_min = 0
        self.clip_max = 2 ** 24 - 1

    def __call__(self, rgb):
        rgb = rgb.to(torch.float64)
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        mean_r = r.mean(dim=(1, 2), keepdim=True)
        mean_g = g.mean(dim=(1, 2), keepdim=True)
        mean_b = b.mean(dim=(1, 2), keepdim=True)
        r = r * (mean_g / mean_r)
        b = b * (mean_g / mean_b)
        rgb = torch.stack([r, g, b], dim=1)
        rgb = torch.clamp(rgb, self.clip_min, self.clip_max)
        rgb = rgb.float()
        return rgb


class Demosaic(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 convolutions.

    Compared to Debayer2x2 this method does not use upsampling.
    Instead, we identify five 3x3 interpolation kernels that
    are sufficient to reconstruct every color channel at every
    pixel location.

    We convolve the image with these 5 kernels using stride=1
    and a one pixel reflection padding. Finally, we gather
    the correct channel values for each pixel location. Todo so,
    we recognize that the Bayer pattern repeats horizontally and
    vertically every 2 pixels. Therefore, we define the correct
    index lookups for a 2x2 grid cell and then repeat to image
    dimensions.
    """

    def __init__(self):
        super(Demosaic, self).__init__()
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.Tensor([
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25],
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0],
        ]).view(5, 1, 3, 3),
            requires_grad=False,
        ).cuda()
        # fmt: on

        self.index =torch.tensor([
            [0, 3], [4, 2],
            [1, 0], [0, 1],
            [2, 4], [3, 0],
        ],dtype=torch.int64).view(1, 3, 2, 2).cuda()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x/1000.0
        xpad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        c = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        c = torch.cat((c, x), 1)
        c = c*1000.0
        c = torch.tensor(c, dtype=torch.int64)
        # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather(
            c,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand in batch is faster than repeat
        )
        return rgb