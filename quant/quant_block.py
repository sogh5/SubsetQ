#
# MIT License
#
# Copyright (c) 2022 Sangyun Oh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# Our source codes are based on implementation: https://github.com/yhhhli/BRECQ (under MIT License).
# Therefore, open source software under specific license described below (BRECQ) may be contained in our implementation.
#
# BRECQ (Pytorch implementation of BRECQ, ICLR 2021)
# - URL: https://github.com/yhhhli/BRECQ
# - Copyright notice: Copyright (c) 2021 Yuhang Li
# - License: MIT License
# - License notice: refer to the file, "ex_lics/BRECQ-LICENSE"
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.quant_layer import QuantModule, StraightThrough
from models.resnet import Bottleneck

class BaseQuantBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant)

class QuantBottleneck(BaseQuantBlock):
    def __init__(self, bottleneck: Bottleneck, qparams: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.conv1, qparams)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, qparams)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, qparams)
        self.activation_function = bottleneck.relu3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], qparams)
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        return out

specials = {
    Bottleneck: QuantBottleneck,
}
