from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)   # [0,1,2,3]  size(1) = 2 取得是维度上的值
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        # right
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, net_size):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], 10)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x,addf):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        f2 = self.layer1(out)
        if addf !=None :
            addf = F.interpolate(addf, f2.size()[2:], mode='bilinear', align_corners=True)
            f2 = f2 + torch.nn.functional.pad(addf,
                                            [0, 0, 0, 0, 0, f2.size(1) - addf.size(1)],
                                            mode='constant', value=0)

        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = F.relu(self.bn2(self.conv2(f4)))

        return f2,f3,f4,f5


configs = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


def test():
    net = ShuffleNetV2(net_size=0.5)
    x = torch.randn(3, 3, 32, 32)
    y = net(x)
    print(y.shape)





#
# class ShuffleNetV2(nn.Module):
#     """ShuffleNetV2 implementation.
#     """
#
#     def __init__(self, scale=1.0, in_channels=3, c_tag=0.5, num_classes=1000, activation=nn.ReLU,
#                  SE=False, residual=False, groups=2):
#         """
#         ShuffleNetV2 constructor
#         :param scale:
#         :param in_channels:
#         :param c_tag:
#         :param num_classes:
#         :param activation:
#         :param SE:
#         :param residual:
#         :param groups:
#         """
#
#         super(ShuffleNetV2, self).__init__()
#
#         self.scale = scale
#         self.c_tag = c_tag
#         self.residual = residual
#         self.SE = SE
#         self.groups = groups
#
#         self.activation_type = activation
#         self.activation = activation(inplace=True)
#         self.num_classes = num_classes
#
#         self.num_of_channels = {0.5: [24, 48, 96, 192, 1024], 1: [24, 116, 232, 464, 1024],
#                                 1.5: [24, 176, 352, 704, 1024], 2: [24, 244, 488, 976, 2048]}
#         self.c = [_make_divisible(chan, groups) for chan in self.num_of_channels[scale]]
#         self.n = [3, 8, 3]  # TODO: should be [3,7,3]
#         self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(self.c[0])
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.shuffles = self._make_shuffles()
#
#         self.conv_last = nn.Conv2d(self.c[-2], self.c[-1], kernel_size=1, bias=False)
#         self.bn_last = nn.BatchNorm2d(self.c[-1])
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(self.c[-1], self.num_classes)
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def _make_stage(self, inplanes, outplanes, n, stage):
#         modules = OrderedDict()
#         stage_name = "ShuffleUnit{}".format(stage)
#
#         # First module is the only one utilizing stride
#         first_module = DownsampleUnit(inplanes=inplanes, activation=self.activation_type, c_tag=self.c_tag,
#                                       groups=self.groups)
#         modules["DownsampleUnit"] = first_module
#         second_module = BasicUnit(inplanes=inplanes * 2, outplanes=outplanes, activation=self.activation_type,
#                                   c_tag=self.c_tag, SE=self.SE, residual=self.residual, groups=self.groups)
#         modules[stage_name + "_{}".format(0)] = second_module
#         # add more LinearBottleneck depending on number of repeats
#         for i in range(n - 1):
#             name = stage_name + "_{}".format(i + 1)
#             module = BasicUnit(inplanes=outplanes, outplanes=outplanes, activation=self.activation_type,
#                                c_tag=self.c_tag, SE=self.SE, residual=self.residual, groups=self.groups)
#             modules[name] = module
#
#         return nn.Sequential(modules)
#
#     def _make_shuffles(self):
#         modules = OrderedDict()
#         stage_name = "ShuffleConvs"
#
#         for i in range(len(self.c) - 2):
#             name = stage_name + "_{}".format(i)
#             module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i], stage=i)
#             modules[name] = module
#
#         return nn.Sequential(modules)
#
#     def forward(self, x,addf=None):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.activation(x)
#         x = self.maxpool(x)   #torch.Size([2, 24, 79, 119])
#         f2 = self.shuffles[0](x)
#         if addf !=None :
#             addf = F.interpolate(addf, f2.size()[2:], mode='bilinear', align_corners=True)
#             f2 = f2 + torch.nn.functional.pad(addf,
#                                             [0, 0, 0, 0, 0, f2.size(1) - addf.size(1)],
#                                             mode='constant', value=0)
#
#         f3 = self.shuffles[1](f2)
#         f4 = self.shuffles[2](f3)
#
#         # x = self.shuffles(x) #torch.Size([2, 464, 10, 15])
#         x = self.conv_last(f4)
#         x = self.bn_last(x)
#         f5 = self.activation(x)
#
#
#         return f2,f3,f4,f5
#
# if __name__ == "__main__":
#     """Testing
#     """
#     model1 = ShuffleNetV2()
#     print(model1)
#     model2 = ShuffleNetV2(scale=0.5, in_channels=3, c_tag=0.5, num_classes=1000, activation=nn.ReLU,
#                           SE=False, residual=False)
#     print(model2)
#     model3 = ShuffleNetV2(in_channels=2, num_classes=10)
#     print(model3)
#     x = torch.randn(1, 2, 224, 224)
#     print(model3(x))
#     model4 = ShuffleNetV2( num_classes=10, groups=3, c_tag=0.2)
#     print(model4)
#     model4_size = 769
#     x2 = torch.randn(1, 3, model4_size, model4_size, )
#     print(model4(x2))
#     model5 = ShuffleNetV2(scale=2.0,num_classes=10, SE=True, residual=True)
#     x3 = torch.randn(1, 3, 196, 196)
#     print(model5(x3))
#     torch.save(model1,"model1.pth")
#     torch.save(model2,"model2.pth")
#     torch.save(model3, "model3.pth")
#     torch.save(model4, "model4.pth")
