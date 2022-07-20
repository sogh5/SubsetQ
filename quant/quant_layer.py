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


import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import ext_vals as ev
from typing import Union

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def cal_norm(x, y, p=2):
    dist = torch.abs(x) - torch.abs(y)
    return torch.norm(dist, p)

def cand_assign(x):
    x = torch.Tensor(list(set(x)))
    return x.mul(1.0 / torch.max(x)).cuda()

def subset_quant(wgt, qps):
    w_rs  = torch.reshape(wgt, [-1])
    w_q = qps[torch.min(torch.abs((torch.abs(w_rs)[:, None]).expand(len(w_rs), len(qps)) - (qps[None, :]).expand(len(w_rs), len(qps))), 1)[1]]
    return (torch.reshape(torch.mul(w_q, torch.sign(w_rs)), torch.tensor(wgt.size()).tolist()) - wgt).detach() + wgt

def alpha_discover(weight, subset, init=1.0):
    alpha = init
    prevloss = 0
    firstloss = 0
    w = weight.cuda()
    subset = subset.cuda()
    for i in range(1000):
        subseta = torch.mul(subset, alpha)
        qa = subset_quant(w, subseta) 
        loss1 = torch.norm(w - qa, 2) 
        q2 = torch.sum(torch.mul(qa/alpha, qa/alpha))
        qw = torch.sum(torch.mul(qa/alpha, w))
        if q2 == 0: 
            alpha2 = torch.mean(torch.abs(w))
        else:
            alpha2 = torch.div(qw, q2)
        alpha = alpha2
        if torch.abs(prevloss - loss1) < 1e-5: 
            break
        prevloss = loss1
        if i == 0:
            firstloss = loss1
    return alpha

class QPSQuantizer(nn.Module):
    def __init__(self, layer_id = -1, orgm = None, n_bits = 4):
        super(QPSQuantizer, self).__init__()
        assert 2 <= n_bits <= 4, 'bitwidth not supported'
        self.mname = 'QPSQuantizer'
        self.om = orgm
        self.layer_id = layer_id
        self.q_levels = 2 ** (n_bits-1)
        self.qps = None
        self.alpha = None
        self.wm = None
        self.bpath = './QPS_data/'+ev.arch+'/'+'w{}b'.format(n_bits)+'/'
        self.clv = None
        self.calpha = None
        self.sq_init()
        self.inited = True

    def forward(self, x: torch.Tensor):
        xs = torch.div(x, self.wm)
        x_q = self.f_process(self.quantize(xs.mul(self.clv)), self.calpha)
        return x_q.mul(self.wm)

    def quantize(self, x):
        return subset_quant(x, torch.mul(self.qps, self.alpha))

    def qps_discovery(self, x: torch.Tensor):
        base = self.bpath + 'L{}'.format(self.layer_id)
        qpath = base+'.qps'
        apath = base+'.alpha'
        if os.path.exists(qpath):
            with open(qpath, "rb") as fr:
                self.qps = pickle.load(fr)
            with open(apath, "rb") as fr:
                self.alpha = pickle.load(fr)
        else:
            self.qps, self.alpha = self.qdiscovery(x) 
            with open(qpath, "wb") as fw:
                pickle.dump(self.qps, fw)
            with open(apath, "wb") as fw:
                pickle.dump(self.alpha, fw)
        self.inited = True
        return

    def sq_init(self):
        os.makedirs(self.bpath, exist_ok=True)
        x = self.om.weight
        self.wm = torch.max(torch.abs(x))
        self.qps_discovery(torch.div(x, self.wm).data.clone().detach())
        self.clv = nn.Parameter(torch.ones(x.size()))
        self.calpha = nn.Parameter(torch.ones(self.om.out_channels, 1))
        print('SQ layer initialized (id: {}, QPS: {})'.format(self.layer_id, self.qps.data.cpu()))

    def f_process(self, x, sv):
        d = x.size()
        return torch.mul(x.view(d[0], -1), sv.expand(d[0], d[1]*d[2]*d[3]).cuda()).view(d)

    def qdiscovery(self, x):
        return self.qsearch(x, ev.qps_pool)

    def qsearch(self, x, pool):
        floss = 9999
        fqi = -1
        for qi in range(len(pool)):
            qps = cand_assign(pool[qi])
            alpha = alpha_discover(x, qps)
            qps = torch.mul(qps, alpha)
            qval = subset_quant(x, qps)
            dist = cal_norm(x, qval, 2)
            if floss > dist:
                floss = dist
                fqi = qi
        fqps = cand_assign(pool[fqi])
        alpha = alpha_discover(x, fqps)
        return fqps, alpha

class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, \
            channel_wise: bool = False, scale_method: str = 'mse', leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.mname = 'UniformAffineQuantizer'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)

class QuantModule(nn.Module):
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], qparams: dict = {}):
        super(QuantModule, self).__init__()

        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        self.use_quant = False

        if isinstance(org_module, nn.Conv2d) and org_module.in_channels != 3:
            self.weight_quantizer = QPSQuantizer(layer_id=ev.layer_id, orgm=org_module, **qparams)
            ev.layer_id = ev.layer_id + 1
        else:
            self.weight_quantizer = UniformAffineQuantizer(**qparams)
            print('Uniform quantizer applied: ', type(org_module))

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        if self.use_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        return self.activation_function(out)

    def set_quant_state(self, use_q: bool = False):
        self.use_quant = use_q
