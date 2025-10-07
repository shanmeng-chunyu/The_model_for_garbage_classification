import torch
import torch.nn as nn

#It's coding by myself
#So it's without pretraining
#So it's unefficient
#Let's drop it


# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, expand_ratio, stride=1, ):
#         super().__init__()
#         self.stride = stride
#         hidden_channels = in_channels * expand_ratio
#         self.use_res_connect = self.stride == 1 and in_channels == out_channels
#         layers = []
#         if expand_ratio != 1:
#             layers.extend([
#                 nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0,
#                           bias=False),
#                 nn.BatchNorm2d(hidden_channels),
#                 nn.ReLU6(inplace=True)
#             ])
#         layers.extend([
#             nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=stride,
#                       groups=hidden_channels, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_channels),
#             nn.ReLU6(inplace=True)
#         ])
#         layers.extend([
#             nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
#                       bias=False),
#             nn.BatchNorm2d(out_channels),
#         ])
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.conv2d1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU6(inplace=True),
#         )
#         self.bottleneck = nn.Sequential(
#             self.generate_block(in_channels=32, out_channels=16, expand_ratio=1, num_blocks=1, stride=1),
#             self.generate_block(in_channels=16, out_channels=24, expand_ratio=6, num_blocks=2, stride=2),
#             self.generate_block(in_channels=24, out_channels=32, expand_ratio=6, num_blocks=3, stride=2),
#             self.generate_block(in_channels=32, out_channels=64, expand_ratio=6, num_blocks=4, stride=2),
#             self.generate_block(in_channels=64, out_channels=96, expand_ratio=6, num_blocks=3, stride=1),
#             self.generate_block(in_channels=96, out_channels=160, expand_ratio=6, num_blocks=3, stride=2),
#             self.generate_block(in_channels=160, out_channels=320, expand_ratio=6, num_blocks=1, stride=1),
#         )
#         self.conv2d2 = nn.Sequential(
#             nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(1280),
#             nn.ReLU6(inplace=True),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.conv2d3 = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Conv2d(in_channels=1280, out_channels=num_classes, kernel_size=1, stride=1)
#         )
#
#     def generate_block(self, in_channels, out_channels, expand_ratio, num_blocks, stride=1):
#         layers = []
#         layers.extend([
#             Bottleneck(in_channels=in_channels, out_channels=out_channels, expand_ratio=expand_ratio, stride=stride),
#         ])
#         for i in range(num_blocks - 1):
#             layers.append(
#                 Bottleneck(in_channels=out_channels, out_channels=out_channels, expand_ratio=expand_ratio, stride=1))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv2d1(x)
#         x = self.bottleneck(x)
#         x = self.conv2d2(x)
#         x = self.avgpool(x)
#         x = self.conv2d3(x)
#         x = torch.flatten(x, 1)
#         return x
