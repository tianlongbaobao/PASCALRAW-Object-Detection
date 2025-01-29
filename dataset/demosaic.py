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


def demosaic(raw):
    C, H, W = raw.shape

    # 重新初始化一个空的 RGB 图像
    rgb = np.zeros((3, H, W))

    # 假设 Bayer 模式是 RGGB

    # 红色通道 (偶数行、偶数列)
    rgb[0, ::2, ::2] = raw[0, ::2, ::2]
    rgb[0, 1::2, 1::2] = raw[0, ::2, ::2]
    rgb[0, 1:-1:2, ::2] = (rgb[0, :-2:2, ::2] + rgb[0, 2::2, ::2])/2
    rgb[0, -1, ::2] = rgb[0, -2, ::2]
    rgb[0, ::2, 1:-1:2] = (rgb[0, ::2, :-2:2] + rgb[0, ::2, 2::2]) / 2
    rgb[0, ::2, -1] = rgb[0, ::2, -2]


    # 蓝色通道 (奇数行、奇数列)
    rgb[2, ::2, ::2] = raw[0, 1::2, 1::2]
    rgb[2, 1::2, 1::2] = raw[0, 1::2, 1::2]  # 红色像素直接来自原始 Bayer
    rgb[2, 1:-1:2, ::2] = (rgb[2, :-2:2, ::2] + rgb[2, 2::2, ::2]) / 2
    rgb[2, -1, ::2] = rgb[2, -2, ::2]
    rgb[2, ::2, 1:-1:2] = (rgb[2, ::2, :-2:2] + rgb[2, ::2, 2::2]) / 2
    rgb[2, ::2, -1] = rgb[2, ::2, -2]  # 蓝色像素直接来自原始 Bayer

    # 通过插值恢复绿色像素
    rgb[1, 1::2, ::2] = raw[0, 1::2, ::2]
    rgb[1, ::2, 1::2] = raw[0, ::2, 1::2]
    rgb[1, 1:-1:2, 1::2] = (rgb[1, :-2:2, 1::2] + rgb[1, 2::2, 1::2]) / 2
    rgb[1, -1, 1::2] = rgb[1, -2, 1::2]
    rgb[1, ::2, 2::2] = (rgb[1, ::2, 1:-1:2] + rgb[1, ::2, 3::2]) / 2
    rgb[1, ::2, 0] = rgb[0, ::2, 1]
    return rgb

def rgbraw(rgb):
    B, C, H, W = rgb.shape

    # 初始化一个空的张量用于保存恢复后的 Bayer 图像（原始只有一个通道）
    raw_bayer = np.zeros((B, 1, H, W))

    # 假设 Bayer 模式是 RGGB
    # 绿色像素是通过插值得到的，我们需要用两边的绿色像素来计算其平均值。

    # 红色像素 (偶数行、偶数列)
    raw_bayer[:, 0, ::2, ::2] = rgb[:, 0, ::2, ::2]  # 红色通道 (R) 直接来自于红色通道

    # 绿色像素 (偶数行、奇数列 和 奇数行、偶数列)
    raw_bayer[:, 0, ::2, 1::2] = rgb[:, 0, ::2, 1::2] # 绿色通道 (G) 是两侧绿色像素的平均值
    raw_bayer[:, 0, 1::2, ::2] = rgb[:, 0, 1::2, ::2] # 绿色通道 (G) 是两侧绿色像素的平均值

    # 蓝色像素 (奇数行、奇数列)
    raw_bayer[:, 0, 1::2, 1::2] = rgb[:, 2, 1::2, 1::2]  # 蓝色通道 (B) 直接来自于蓝色通道

    return raw_bayer

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