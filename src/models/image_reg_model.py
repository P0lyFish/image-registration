import torch
import models.cupy_module.adacof as adacof
from torch.nn import functional as F
import torch.nn as nn
from utils.training_utils import CharbonnierFunc, moduleNormalize


class ImageRegModel(torch.nn.Module):
    def __init__(self, args):
        super(ImageRegModel, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = PrunedKernelEstimation(self.kernel_size)

        self.context_synthesis = GridNet(63, 3)  # (in_channel, out_channel) = (126, 3) for the synthesis network

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frames):
        source, target = frames['src'], frames['target']

        if source.shape != target.shape:
            raise ValueError('Source and target images must have the same size!')

        h = int(list(source.size())[2])
        w = int(list(source.size())[3])

        h_padded = False
        w_padded = False
        if h % 32 != 0:
            pad_h = 32 - (h % 32)
            source = F.pad(source, [0, 0, 0, pad_h], mode='reflect')
            target = F.pad(target, [0, 0, 0, pad_h], mode='reflect')
            h_padded = True

        if w % 32 != 0:
            pad_w = 32 - (w % 32)
            source = F.pad(source, [0, pad_w, 0, 0], mode='reflect')
            target = F.pad(target, [0, pad_w, 0, 0], mode='reflect')
            w_padded = True

        Weight, Alpha, Beta, featConv1, featConv2, featConv3, featConv4, featConv5 \
            = self.get_kernel(moduleNormalize(source), moduleNormalize(target))

        tensorAdaCoF = self.moduleAdaCoF(self.modulePad(source), Weight, Alpha, Beta, self.dilation) * 1.

        w, h = self.modulePad(source).shape[2:]

        tensorConv1_ = F.interpolate(featConv1, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv1L = self.moduleAdaCoF(tensorConv1_, Weight, Alpha, Beta, self.dilation) * 1.

        tensorConv2_ = F.interpolate(featConv2, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv2L = self.moduleAdaCoF(tensorConv2_, Weight, Alpha, Beta, self.dilation) * 1.

        tensorConv3_ = F.interpolate(featConv3, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv3L = self.moduleAdaCoF(tensorConv3_, Weight, Alpha, Beta, self.dilation) * 1.

        tensorConv4_ = F.interpolate(featConv4, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv4L = self.moduleAdaCoF(tensorConv4_, Weight, Alpha, Beta, self.dilation) * 1.

        tensorConv5_ = F.interpolate(featConv5, size=(w, h), mode='bilinear', align_corners=False)
        tensorConv5L = self.moduleAdaCoF(tensorConv5_, Weight, Alpha, Beta, self.dilation) * 1.

        tensorCombined = torch.cat(
            [tensorAdaCoF, tensorConv1L, tensorConv2L,
             tensorConv3L, tensorConv4L, tensorConv5L], dim=1)

        source_warped = self.context_synthesis(tensorCombined)

        if h_padded:
            source_warped = source_warped[:, :, 0:h, :]
        if w_padded:
            source_warped = source_warped[:, :, :, 0:w]

        if self.training:
            # Smoothness Terms
            m_Alpha = torch.mean(Weight * Alpha, dim=1, keepdim=True)
            m_Beta = torch.mean(Weight * Beta, dim=1, keepdim=True)

            g_Alpha = CharbonnierFunc(m_Alpha[:, :, :, :-1] - m_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                m_Alpha[:, :, :-1, :] - m_Alpha[:, :, 1:, :])
            g_Beta = CharbonnierFunc(m_Beta[:, :, :, :-1] - m_Beta[:, :, :, 1:]) + CharbonnierFunc(
                m_Beta[:, :, :-1, :] - m_Beta[:, :, 1:, :])

            g_Spatial = g_Alpha + g_Beta  # used as total variation loss during training

            return {
                'pred': source_warped,
                'regs': {
                    'GSpatialReg': g_Spatial
                }
            }
        else:
            return {'pred': source_warped}


class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96)):
        super(GridNet, self).__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)


class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LateralBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x


class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.f(x)


class PrunedKernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(PrunedKernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        self.module1by1_1 = torch.nn.Conv2d(in_channels=24, out_channels=4, kernel_size=1, stride=1, padding=1)
        self.module1by1_2 = torch.nn.Conv2d(in_channels=52, out_channels=8, kernel_size=1, stride=1, padding=1)
        self.module1by1_3 = torch.nn.Conv2d(in_channels=95, out_channels=12, kernel_size=1, stride=1, padding=1)
        self.module1by1_4 = torch.nn.Conv2d(in_channels=159, out_channels=16, kernel_size=1, stride=1, padding=1)
        self.module1by1_5 = torch.nn.Conv2d(in_channels=121, out_channels=20, kernel_size=1, stride=1, padding=1)

        self.moduleConv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=48, out_channels=52, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=52, out_channels=99, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=99, out_channels=97, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=97, out_channels=95, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=95, out_channels=156, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=156, out_channels=142, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=142, out_channels=159, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=159, out_channels=92, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=92, out_channels=72, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=72, out_channels=121, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=121, out_channels=99, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=99, out_channels=69, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=69, out_channels=36, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=36, out_channels=121, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=121, out_channels=74, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=74, out_channels=83, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=83, out_channels=81, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=81, out_channels=159, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=159, out_channels=83, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=83, out_channels=88, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=88, out_channels=72, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=72, out_channels=95, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=95, out_channels=45, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=45, out_channels=45, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=45, out_channels=45, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=45, out_channels=52, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleWeight = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1),
            torch.nn.Softmax(dim=1)
        )
        self.moduleAlpha = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1)
        )
        self.moduleBeta = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=52, out_channels=49, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=21, out_channels=121, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, rfield0, rfield1):
        tensorJoin = torch.cat([rfield0, rfield1], 1)
        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        Weight = self.moduleWeight(tensorCombine)
        Alpha = self.moduleAlpha(tensorCombine)
        Beta = self.moduleBeta(tensorCombine)

        featConv1 = self.module1by1_1(tensorConv1)
        featConv2 = self.module1by1_2(tensorConv2)
        featConv3 = self.module1by1_3(tensorConv3)
        featConv4 = self.module1by1_4(tensorConv4)
        featConv5 = self.module1by1_5(tensorConv5)

        return Weight, Alpha, Beta, featConv1, featConv2, featConv3, featConv4, featConv5
