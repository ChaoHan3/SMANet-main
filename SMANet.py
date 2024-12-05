import math
import numbers
import torch
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from torchvision import ops
from einops.layers.torch import Rearrange


def make_layer(basic_block, num_basic_block, **kwarg):  # 用于创建由相同基本块组成的层，并将这些基本块堆叠起来
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
        **kwarg 允许传入额外的关键字参数，这些参数将传递给基本块的构造函数
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):  # 循环迭代几次
        layers.append(basic_block(**kwarg))  # 每次迭代中，通过basic_来构造函数，并传入**kwarg中任何额外参数，创建一个新的基本块实例，并添加到layers列表中
    return nn.Sequential(*layers)  # 将所有块堆叠在一起并返回。


@ARCH_REGISTRY.register()
class SMANet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=8, upscale=2, res_scale=1.0):
        super(SMANet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # 用于创建多个残差块，其中每个残差块由ResidualBlock类组成。num_block参数指定了创建的残差块数量，而num_feat和res_scale将参数传递给ResidualBlock类的初始化方法
        self.body = make_layer(ResidualBlock, num_block, num_feat=num_feat, res_scale=res_scale)
        # 这个卷积层将num_feat个特征的图像转换为另外num_feat个特征的推向，其卷积核大小为3*3，步长为1，填充为1

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.CGAFusion = CGAFusion(num_feat)

        self.upsample = Upsample(upscale, num_feat)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        res = x
        block_count = 0
        for block in self.body:
            res = block(res)  # 使用 res 而不是 x
            block_count += 1
            if block_count % 2 == 0:
                res = self.CGAFusion(res, x)
                res = self.conv_after_body(res)
                res += x  # 添加残差连接
        x = self.conv_after_body(x)  # 不用 x 会出现错误，但是这里的操作不影响 res
        res += x  # 最后一次残差连接
        x = self.conv_last(self.upsample(res))  # 使用 res 而不是 x
        return x


class ResidualBlock(nn.Module):  # 实现了残差块的功能
    def __init__(self, num_feat=64, res_scale=1, idynamic_ffn_expansion_factor=2, input_resolution=None, num_head=8,
                 chsa_head_ratio=0.25, window_size=8, shift=0, head_dim=None, qkv_bias=True, mv_ver=1,
                 hidden_ratio=2.0, act_layer=nn.GELU, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, helper=True,
                 mv_act=nn.LeakyReLU, exp_factor=1.2,
                 expand_groups=4):  # 特征数量64，res_scale残差比例，默认为1，残差缩放比例的作用是在残差网络中调节残差的重要性，提高模型的性能和训练稳定性。
        super(ResidualBlock, self).__init__()
        # 用于控制残差的缩放比例
        self.baseblock1 = Baseblock1(num_feat)

        self.baseblock7 = Baseblock7(num_feat, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=False,
                                     input_resolution=input_resolution)

        self.norm2 = LayerNorm(num_feat)

        self.res_scale = res_scale

        self.ESMC = ESMC(num_feat)
        self.FFN = FFN(num_feat)

    def forward(self, x):
        identity = x  # 在后续残差中使用

        x = self.norm2(x)

        x1 = self.baseblock1(x)
        x1 = self.baseblock7(x1)

        x2 = self.ESMC(x)

        x = self.FFN(x1 + x2)

        return x * self.res_scale + identity


class Attention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn


class Baseblock7(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """

    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None, res_scale=1):
        super(Baseblock7, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.res_scale = res_scale
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.attention = Attention(dim)

    def forward(self, x):
        identity = x
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = self.attention(x)

        return x * self.res_scale + identity


class Baseblock1(nn.Module):  # (3,3,64,64)>(3,64,64,64)
    def __init__(self, num_feat):
        super(Baseblock1, self).__init__()
        self.num_feat = num_feat
        # 定义第一个低维度分支
        self.low = nn.Sequential(
            nn.Conv2d(num_feat, 32, 3, padding=1),
            nn.GELU()
        )
        # 定义第二个高纬度分支
        self.high = nn.Sequential(
            nn.Conv2d(num_feat, 64, 3, 1, padding=1),  # 修改此处的卷积核大小和填充
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, padding=1)  # 修改此处的卷积核大小和填充
        )
        self.fusion = nn.Conv2d(32 + 128, 64, 3, 1, padding=1)  # 修改此处的输入通道数和卷积核大小


    def forward(self, x):
        identity = x
        x1 = self.low(x)
        x2 = self.high(x)

        combined = torch.cat((x1, x2), dim=1)
        x = self.fusion(combined)

        return x + identity


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class LayerNorm(nn.Module):  # 对输入进行了layernormlization
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):  # 根据date_format选择不同的归一化方式
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


# partial convolution-based feed-forward network
class FFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x


# self-modulation feature aggregation (SMFA) module
class ESMC(nn.Module):
    def __init__(self, dim=36):
        super(ESMC, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        # y_d = self.lde(y)
        return self.linear_2(x_l)
