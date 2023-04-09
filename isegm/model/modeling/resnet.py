import torch
from isegm.model.modeling.resnetv1b import resnet34_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s


class ResNetBackbone(torch.nn.Module):
    def __init__(self, backbone='resnet50', pretrained_base=False, dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()
        pretrained_base = False  ## TODO 需要需改回来
        if backbone == 'resnet34':
            pretrained = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet50':
            pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError(f'unknown backbone: {backbone}')

        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


#
# model  = ResNetBackbone()
#
# input_data = torch.rand(1, 3,255,255)
#
# result1 = model(input_data)
#1
# print(result1)