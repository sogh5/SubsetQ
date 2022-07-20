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
from quant.quant_layer import QuantModule, StraightThrough, lp_loss
from quant.quant_model import QuantModel
from quant.quant_block import BaseQuantBlock
from quant.data_utils import save_grad_data, save_inp_oup_data
import ext_vals as ev

def block_reconstruction(model: QuantModel, block: BaseQuantBlock, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = True, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False):

    model.set_quant_state(False)
    block.set_quant_state(True)

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    opt_params = []
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            opt_params += [module.weight_quantizer.clv]
            opt_params += [module.weight_quantizer.calpha]
    optimizer = torch.optim.Adam(opt_params, lr=ev.qps_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_func = QPSLossFunction(block, weight=weight, max_count=iters, rec_loss=opt_mode,
                                b_range=b_range, decay_start=0, warmup=warmup, p=p)

    cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, asym, False, batch_size)
    device = 'cuda'

    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)

        optimizer.zero_grad()
        out_quant = block(cur_inp)

        err = loss_func(out_quant, cur_out, None)
        err.backward(retain_graph=True)

        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    if not include_act_func:
        block.activation_function = org_act_func

class QPSLossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.qps_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start, start_b=1e-2, end_b=1e-3)
        self.count = 0

    def cal_norm(self, x, y, p=2):
        dist = torch.abs(x) - torch.abs(y)
        return torch.norm(dist, p)

    def __call__(self, pred, tgt, grad=None):
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)
        qd = self.qps_decay(self.count)
        trace = None 

        if self.count < self.loss_start or self.round_loss == 'none':
            qps_loss = 0
        elif self.round_loss == 'relaxation': 
            qps_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    trace = module.weight_quantizer.calpha
                    qps_lval = torch.mul(self.cal_norm(module.weight, module.weight_quantizer(x=module.weight), 2), qd)
                    qps_loss += qps_lval
        else:
            raise NotImplementedError

        if self.count % 1000 == 0:
            print('QPS loss:{:.3f})\titer={}\tcalpha_mean={:.4f}'.format(float(qps_loss), self.count, 0.0 if trace is None else trace.mean()))
        return rec_loss + qps_loss

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

