#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, num_classes=1000):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_1  = InvertedResidual(64, 64, 2, False, 2)
        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1  = InvertedResidual(64, 64, 2, False, 2)

        self.conv5_1  = InvertedResidual(64, 64, 1, False, 4)
        self.block5_2 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_3 = InvertedResidual(64, 64, 1, True, 4)

        self.conv6_1  = InvertedResidual(64, 32, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(32, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 64, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(64)
        self.relu8 = nn.ReLU(inplace=True)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)

        self.fc0 = nn.Linear(2368, 128)
        self.fc1 = nn.Linear(128, num_classes)


    def forward(self, x):  # x: 3, 112, 112
        x = self.relu1(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu2(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        out1 = self.block3_3(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1).contiguous()

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1).contiguous()

        x3 = self.relu8(self.bn8(self.conv8(x)))
        x3 = x3.view(x3.size(0), -1).contiguous()
        
        multi_scale = torch.cat([x1, x2], dim=1)
        multi_scale = torch.cat([multi_scale, x3], dim=1)

        #print(multi_scale.shape)

        out = self.fc0(multi_scale)
        out = self.fc1(out)

        return torch.sigmoid(out)
        #return torch.softmax(out , dim=1)

