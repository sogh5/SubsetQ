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


import torch.nn as nn
from quant.quant_block import specials, BaseQuantBlock
from quant.quant_layer import QuantModule, StraightThrough
from quant.fold_bn import search_fold_and_remove_bn

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, qparams: dict = {}):
        super().__init__()
        search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(self.model, qparams)

    def quant_module_refactor(self, module: nn.Module, qparams: dict = {}):
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, qparams))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, qparams))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, qparams)

    def set_quant_state(self, quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(quant)

    def forward(self, input):
        return self.model(input)

    def show_qps(self):
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                if m.weight_quantizer.mname == 'QPSQuantizer':
                    print('SQ layer {}] QPS: {}'.format(m.weight_quantizer.layer_id, m.weight_quantizer.qps.cpu().data))

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[0].ignore_reconstruction = True
        module_list[-1].ignore_reconstruction = True

