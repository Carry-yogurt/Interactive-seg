# import torch
# import torch.nn as nn
#
# from isegm.model.ops import DistMaps
# from contextlib import ExitStack
#
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# from isegm.model.modeling.basic_blocks import SeparableConv2d
# from isegm.model.modeling.resnet import ResNetBackbone
# from isegm.model import ops
# # 骨干网络是eff 的deeplab
# class DeepLabV3Plus(nn.Module):
#     def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d,
#                  backbone_norm_layer=None,
#                  ch=256,
#                  project_dropout=0.5,
#                  inference_mode=False,
#                  **kwargs):
#         super(DeepLabV3Plus, self).__init__()
#         if backbone_norm_layer is None:
#             backbone_norm_layer = norm_layer
#
#         # self.backbone_name = backbone
#         self.norm_layer = norm_layer
#         self.backbone_norm_layer = backbone_norm_layer
#         # 根据这个判断是否反向传播
#         self.inference_mode = False
#         # 不知道这个ch用来做什么
#         self.ch = ch
#         self.aspp_in_channels = 2048
#
#         self.skip_project_in_channels = 256  # layer 1 out_channels
#
#         self._kwargs = kwargs
#         if backbone == 'resnet34':
#             self.aspp_in_channels = 512
#             self.skip_project_in_channels = 64
#
#         self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False,
#                                        norm_layer=self.backbone_norm_layer, **kwargs)
#
#         self.head = _DeepLabHead(in_channels=ch + 32, mid_channels=ch, out_channels=ch,
#                                  norm_layer=self.norm_layer)
#
#         self.skip_project = _SkipProject(self.skip_project_in_channels, 32, norm_layer=self.norm_layer)
#         self.aspp = _ASPP(in_channels=self.aspp_in_channels,
#                           atrous_rates=[12, 24, 36],
#                           out_channels=ch,
#                           project_dropout=project_dropout,
#                           norm_layer=self.norm_layer)
#
#         if inference_mode:
#             self.set_prediction_mode()
#
#     def load_pretrained_weights(self):
#         pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True,
#                                     norm_layer=self.backbone_norm_layer, **self._kwargs)
#         backbone_state_dict = self.backbone.state_dict()
#         pretrained_state_dict = pretrained.state_dict()
#
#         backbone_state_dict.update(pretrained_state_dict)
#         self.backbone.load_state_dict(backbone_state_dict)
#
#         if self.inference_mode:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
#
#     def set_prediction_mode(self):
#         self.inference_mode = True
#         self.eval()
#
#     def forward(self, x):
#         with ExitStack() as stack:
#             #TODO 看什么时候停起梯度
#             if self.inference_mode:
#                 stack.enter_context(torch.no_grad())
#             #  拿到resnet 的特征图
#             c1, _, c3, c4 = self.backbone(x)
#             # 拿到resnet输入的一部分特征图 卷了一下
#             c1 = self.skip_project(c1)
#
#             x = self.aspp(c4)
#             # 给aspp输出的结果进行插值 插值后和resnetc1 的大小相同
#             x = F.interpolate(x, c1.size()[2:], mode='bilinear', align_corners=True)
#             x = torch.cat((x, c1), dim=1)
#             x = self.head(x)
#
#         return x,
#
#
# class _SkipProject(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
#         super(_SkipProject, self).__init__()
#         _activation = ops.select_activation_function("relu")
#
#         self.skip_project = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             norm_layer(out_channels),
#             _activation()
#         )
#
#     def forward(self, x):
#         return self.skip_project(x)
#
#
# class _DeepLabHead(nn.Module):
#     def __init__(self, out_channels, in_channels, mid_channels=256, norm_layer=nn.BatchNorm2d):
#         super(_DeepLabHead, self).__init__()
#         #TODO 可以修改一下为传统卷积
#         # 可分离卷积块
#         self.block = nn.Sequential(
#             SeparableConv2d(in_channels=in_channels, out_channels=mid_channels, dw_kernel=3,
#                             dw_padding=1, activation='relu', norm_layer=norm_layer),
#             SeparableConv2d(in_channels=mid_channels, out_channels=mid_channels, dw_kernel=3,
#                             dw_padding=1, activation='relu', norm_layer=norm_layer),
#             nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
#         )
#
#     def forward(self, x):
#         return self.block(x)
#
#
# class _ASPP(nn.Module):
#     def __init__(self, in_channels, atrous_rates, out_channels=256,
#                  project_dropout=0.5, norm_layer=nn.BatchNorm2d):
#         super(_ASPP, self).__init__()
#
#         b0 = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
#             norm_layer(out_channels),
#             nn.ReLU()
#         )
#
#         rate1, rate2, rate3 = tuple(atrous_rates)
#         b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
#         b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
#         b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
#         b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)
#
#         self.concurent = nn.ModuleList([b0, b1, b2, b3, b4])
#
#         project = [
#             nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels,
#                       kernel_size=1, bias=False),
#             norm_layer(out_channels),
#             nn.ReLU()
#         ]
#         if project_dropout > 0:
#             project.append(nn.Dropout(project_dropout))
#         self.project = nn.Sequential(*project)
#
#     def forward(self, x):
#         x = torch.cat([block(x) for block in self.concurent], dim=1)
#f
#         return self.project(x)
#
#
# class _AsppPooling(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer):
#         super(_AsppPooling, self).__init__()
#
#         self.gap = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                       kernel_size=1, bias=False),
#             norm_layer(out_channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         pool = self.gap(x)
#         return F.interpolate(pool, x.size()[2:], mode='bilinear', align_corners=True)
#
#
# def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
#     block = nn.Sequential(
#         nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                   kernel_size=3, padding=atrous_rate,
#                   dilation=atrous_rate, bias=False),
#         norm_layer(out_channels),
#         nn.ReLU()
#     )
#
#     return block
#
#
#
# def get_effec_deep():
#
#
#
#
#
#
# def get_deeplab_model(backbone='resnet50', deeplab_ch=256, aspp_dropout=0.5,
#                       norm_layer=nn.BatchNorm2d, backbone_norm_layer=None,
#                       use_rgb_conv=True, cpu_dist_maps=False,
#                       norm_radius=260):
#     model = DistMapsModel(
#         # 特征抽取 下一步就是可分离卷积
#         feature_extractor=DeepLabV3Plus(backbone=backbone,
#                                         ch=deeplab_ch,
#                                         project_dropout=aspp_dropout,
#                                         norm_layer=norm_layer,
#                                         backbone_norm_layer=backbone_norm_layer),
#         # 分割头
#         head=SepConvHead(1, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
#                          num_layers=2, norm_layer=norm_layer),
#         use_rgb_conv=use_rgb_conv,
#         norm_layer=norm_layer,
#         norm_radius=norm_radius,
#         cpu_dist_maps=cpu_dist_maps
#     )
#     return model
# ## 距离图modle
# class DistMapsModel(nn.Module):
#     def __init__(self, feature_extractor, head, norm_layer=nn.BatchNorm2d, use_rgb_conv=True,
#                  cpu_dist_maps=False, norm_radius=260):
#         super(DistMapsModel, self).__init__()
#         # rgb conv    输入通道为五  输出通道为 3
#         if use_rgb_conv:
#             self.rgb_conv = nn.Sequential(
#                 nn.Conv2d(in_channels=5, out_channels=8, kernel_size=1),
#                 nn.LeakyReLU(negative_slope=0.2),
#                 norm_layer(8),
#                 nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1),
#             )
#         else:
#             self.rgb_conv = None
#         self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
#                                   cpu_mode=cpu_dist_maps)
#         ## 特征抽取阶段
#         self.feature_extractor = feature_extractor
#         # 分割头
#         self.head = head
#     def forward(self, image, points):
#         # 输入图片和点击点 得到 特征图
#         coord_features = self.dist_maps(image, points)
#
#         if self.rgb_conv is not None:
#             x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
#         else:
#             c1, c2 = torch.chunk(coord_features, 2, dim=1)
#             c3 = torch.ones_like(c1)
#             coord_features = torch.cat((c1, c2, c3), dim=1)
#             x = 0.8 * image * coord_features + 0.2 * image
#
#         backbone_features = self.feature_extractor(x)
#         # 使用可分离卷积
#         instance_out = self.head(backbone_features[0])
#
#         instance_out = nn.functional.interpolate(instance_out, size=image.size()[2:],
#                                                  mode='bilinear', align_corners=True)
#         return {'instances': instance_out}
#
#     def load_weights(self, path_to_weights):
#         current_state_dict = self.state_dict()
#         new_state_dict = torch.load(path_to_weights, map_location='cpu')
#         current_state_dict.update(new_state_dict)
#         self.load_state_dict(current_state_dict)
#
#     def get_trainable_params(self):
#         backbone_params = nn.ParameterList()
#         other_params = nn.ParameterList()
#
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 if 'backbone' in name:
#                     backbone_params.append(param)
#                 else:
#                     other_params.append(param)
#         return backbone_params, other_params
#
#
# # 改变原文中的backbone网络试一下
# def get_deeplab_model_seresnet(backbone='se_resnet50', deeplab_ch=256, aspp_dropout=0.5,
#                       norm_layer=nn.BatchNorm2d, backbone_norm_layer=None,
#                       use_rgb_conv=True, cpu_dist_maps=False,
#                       norm_radius=260):
#     model = DistMapsModel(
#         # 特征抽取 下一步就是可分离卷积
#         feature_extractor=DeepLabV3Plus_seResnet(backbone=backbone,
#                                         ch=deeplab_ch,
#                                         project_dropout=aspp_dropout,
#                                         norm_layer=norm_layer,
#                                         backbone_norm_layer=backbone_norm_layer),
#         # 分割头
#         head=SepConvHead(1, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
#                          num_layers=2, norm_layer=norm_layer),
#         use_rgb_conv=use_rgb_conv,
#         norm_layer=norm_layer,
#         norm_radius=norm_radius,
#         cpu_dist_maps=cpu_dist_maps
#     )
#     return model