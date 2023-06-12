import torch.nn as nn
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Tuple
import torch
from torch.nn import functional as F
from torchsummary import summary
from mmcv.cnn import constant_init, kaiming_init


# ----------------------------------InceptionV2 block----------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int,
                 pool_proj: int, conv_block: Optional[Callable[..., nn.Module]] = None, ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch3x3, kernel_size=1),
            conv_block(ch3x3, ch3x3, kernel_size=3, padding=1),
            conv_block(ch3x3, ch3x3, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


# ----------------------------------MCD block----------------------------
def Conv2D_dilation(in_features: int, out_features: int, stride: int, padding: str, dilation: int):
    return nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)


class MDCN(nn.Module):
    def __init__(self, n_filters):
        super(MDCN, self).__init__()
        self.BN_LeakyReLU = nn.Sequential(
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2d_dilation_4 = Conv2D_dilation(n_filters, n_filters, stride=1, padding='same', dilation=4)
        self.conv2d_dilation_3 = Conv2D_dilation(n_filters, n_filters, stride=1, padding='same', dilation=3)
        self.conv2d_dilation_2 = Conv2D_dilation(n_filters, n_filters, stride=1, padding='same', dilation=2)

    def forward(self, x):
        # stream left
        conv_left = self.conv2d_dilation_4(x)
        conv_left = self.BN_LeakyReLU(conv_left)
        # stream_middle_up
        conv_middle_1 = self.conv2d_dilation_3(x)
        conv_middle_1 = self.BN_LeakyReLU(conv_middle_1)
        # steam_right_up
        conv_right_1 = self.conv2d_dilation_2(x)
        conv_right_1 = self.BN_LeakyReLU(conv_right_1)
        conv_right_2 = self.conv2d_dilation_2(conv_right_1)
        conv_right_2 = self.BN_LeakyReLU(conv_right_2)

        # stream_sum_1
        sum_1 = torch.sum(torch.stack([conv_middle_1, conv_right_2]), dim=0)

        # stream_middle_down
        conv_middle_2 = self.conv2d_dilation_3(sum_1)
        conv_middle_2 = self.BN_LeakyReLU(conv_middle_2)
        # stream_right_down
        conv_right_3 = self.conv2d_dilation_2(sum_1)
        conv_right_3 = self.BN_LeakyReLU(conv_right_3)
        conv_right_4 = self.conv2d_dilation_2(conv_right_3)
        conv_right_4 = self.BN_LeakyReLU(conv_right_4)

        # stream_sum_2
        sum_2 = torch.sum(torch.stack([conv_left, conv_middle_2, conv_right_4, x]), dim=0)

        return sum_2


# -------- GC-Net ---------
#  Reference:
# [1] https://arxiv.org/abs/1904.11492  article
# [2] https://github.com/xvjiarui/GCNet  source code
# -------------------------
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)  # ratio = 1/8 or 1/16 ?
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class MDCN_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDCN_Conv, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.block_1 = MDCN(n_filters=out_channels)
        self.block_2 = MDCN(n_filters=out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.block_1(x)
        x = self.block_2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_filters=64, block_num=1):
        super(Encoder, self).__init__()
        # n_filters = 64

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=3, stride=2, padding=1)  # need to calculate the output feature size in each layer.
        self.BN_LeakyReLU_1 = nn.Sequential(
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv_2 = nn.Conv2d(in_channels=n_filters, out_channels=2 * n_filters, kernel_size=3, stride=2, padding=1)
        self.BN_LeakyReLU_2 = nn.Sequential(
            nn.BatchNorm2d(2 * n_filters),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.block_1 = MDCN(n_filters=2 * n_filters)  # in BN_LeakyReLU_2 ->
        self.block_2 = MDCN(n_filters=2 * n_filters)  # in block1 ->

        self.conv_3 = nn.Conv2d(in_channels=2 * n_filters, out_channels=4 * n_filters, kernel_size=3, stride=2, padding=1)  # in block2 ->
        self.BN_LeakyReLU_3 = nn.Sequential(
            nn.BatchNorm2d(4 * n_filters),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.block_3 = MDCN(n_filters=4 * n_filters)  # in BN_LeakyReLU_3 ->
        self.block_4 = MDCN(n_filters=4 * n_filters)  # in block_3 ->

        self.conv_4 = nn.Conv2d(in_channels=4 * n_filters, out_channels=8 * n_filters, kernel_size=3, stride=2, padding=1)  # in block2 ->
        self.BN_LeakyReLU_4 = nn.Sequential(
            nn.BatchNorm2d(8 * n_filters),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.block_5 = MDCN(n_filters=8 * n_filters)  # in BN_LeakyReLU_4 ->
        self.block_6 = MDCN(n_filters=8 * n_filters)  # in block_4 ->

        # non_local block
        self.non_local_block = nn.ModuleList([ContextBlock(inplanes=8 * n_filters, ratio=1. / 16) for _ in range(block_num)])

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.conv_bottom = nn.Conv2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.BN_LeakyReLU_5 = nn.Sequential(
            nn.BatchNorm2d(4 * n_filters),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):

        conv_1 = self.conv_1(x)
        conv_1 = self.BN_LeakyReLU_1(conv_1)  # 64   ##

        conv_2 = self.conv_2(conv_1)
        conv_2 = self.BN_LeakyReLU_2(conv_2)  # 128

        block_1 = self.block_1(conv_2)  # 128
        block_2 = self.block_2(block_1)  # 128  ##

        conv_3 = self.conv_3(block_2)  # 256
        conv_3 = self.BN_LeakyReLU_3(conv_3)  # 256

        block_3 = self.block_3(conv_3)  # 256
        block_4 = self.block_4(block_3)  # 256  ##

        conv_4 = self.conv_4(block_4)  # 512
        conv_4 = self.BN_LeakyReLU_4(conv_4)  # 512

        block_5 = self.block_5(conv_4)  # 512
        x = self.block_6(block_5)  # 512

        # non local block
        for non_local_blk in self.non_local_block:
            x = non_local_blk(x)

        # -------- # for segmentation task
        conv_bottom = self.conv_bottom(x)  # 512
        conv_bottom = self.BN_LeakyReLU_5(conv_bottom)

        return conv_bottom, block_4, block_2, conv_1

    # # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channel=None, scale_factor=2):
        super(DecoderBottleneck, self).__init__()
        if mid_channel == None:
            mid_channel = out_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super(Decoder, self).__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)  # (128 * 8, 128 * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)  # (128 * 4, 128)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))  # (128 * 2, 64)
        self.decoder4_seg = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 4))  # (64, 16) add

        self.decoder4_edge = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 4))  # (64, 16) add
        self.decoder4_edge_ = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                            nn.Conv2d(int(out_channels * 1 / 2), int(out_channels * 1 / 4), kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(int(out_channels * 1 / 4)),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(int(out_channels * 1 / 4), 2, kernel_size=1, stride=1),
                                            )

        self.classifier4seg = nn.Conv2d(int(out_channels * 1 / 4), class_num, kernel_size=1)
        self.classifier4edge = nn.Conv2d(int(out_channels * 1 / 4), 2, kernel_size=1)

    def forward(self, x, x3, x2, x1):
        d1_x = self.decoder1(x, x3)
        d2_x = self.decoder2(d1_x, x2)
        d3_x = self.decoder3(d2_x, x1)
        d4_seg_x = self.decoder4_seg(d3_x)

        # 'norm_add'
        d4_edge_x = self.decoder4_edge(d3_x)
        add_x = torch.add(d4_seg_x, d4_edge_x)
        edge_x = self.classifier4edge(d4_edge_x)
        seg_x = self.classifier4seg(add_x)

        return seg_x, edge_x


class PLC_Net(nn.Module):
    def __init__(self, out_channels=32, class_num=3, block_num=1):
        super(PLC_Net, self).__init__()
        self.encoder = Encoder(n_filters=out_channels, block_num=block_num)
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x3, x2, x1 = self.encoder(x)
        x = self.decoder(x, x3, x2, x1)

        return x
