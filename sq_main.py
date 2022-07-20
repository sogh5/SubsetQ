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
import numpy as np
import argparse
import sys
import os
import random
import time
import pickle
import math
import model_load
import ext_vals as ev
from itertools import combinations
from imagenet import build_imagenet_data
from quant import *

def gen_universal_set(s_size=4):
    base_p1 = [1, 2**(-1), 2**(-3), 0]
    base_p2 = [1, 2**(-2), 2**(-4), 0]
    uset_p = []
    for i in range(s_size):
        for j in range(s_size):
            pv = (base_p1[i] + base_p2[j])/2
            ap_flag = True
            for k in range(len(uset_p)):
                if pv == uset_p[k]:
                    ap_flag = False
            if ap_flag:
                uset_p.append(pv)
    return uset_p

def gen_sq(args):
    su = gen_universal_set()
    qpoints = int(math.pow(2.0, args.wbits-1))
    sq = list(combinations(su, qpoints))
    print('universal set: {} (size: {}, qps pool size: {})'.format(su, len(su), len(sq)))
    return sq

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subset Quantization', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', default='quant', type=str, choices=['quant', 'test'])
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--data_path', default='', type=str, required=True)
    parser.add_argument('--wgt_load', default='', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--wbits', default=4, type=int, help='bit-precision for weight')
    parser.add_argument('--qps_lr', default=0.0001, type=float, help='lr for QPS optimizer (calib set)')

    args = parser.parse_args()

    ev.arch = args.arch
    ev.qps_pool = gen_sq(args)
    ev.gpuid = args.gpu
    ev.wbits = args.wbits
    ev.qps_lr = args.qps_lr

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    rseed = int(torch.randint(1, 9999, (1,)))
    seed_all(rseed)
    train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers, data_path=args.data_path)
    if args.mode == 'test':
        print('For test, loading: ', args.wgt_load)
        with open(args.wgt_load, "rb") as fr:
            qnn = pickle.load(fr)
        qnn.show_qps()
        qnn.set_quant_state(True)
        wq_acc = validate_model(test_loader, qnn)
        print('Weight quantization accuracy: {}'.format(wq_acc))
        sys.exit(-1)

    cnn = eval('model_load.{}(pretrained=True)'.format(args.arch))
    cnn.cuda()
    cnn.eval()

    qnn = QuantModel(model=cnn, qparams={'n_bits': args.wbits})
    qnn.cuda()
    qnn.eval()
    qnn.set_first_last_layer_to_8bit()

    cali_data = get_train_samples(train_loader, num_samples=1024)
    device = next(qnn.parameters()).device
    qnn.set_quant_state(True)
    _ = qnn(cali_data[:64].to(device))

    kwargs = dict(cali_data=cali_data, warmup=0.2)

    def qp_cali_scale(model: nn.Module):
        for name, module in model.named_children():
            if isinstance(module, BaseQuantBlock):
                print('Quantization parameter optimization {}'.format(name))
                block_reconstruction(qnn, module, **kwargs)
            else:
                qp_cali_scale(module)

    qp_cali_scale(qnn)
    qnn.set_quant_state(True)
    wq_acc = validate_model(test_loader, qnn)
    print('Weight quantization accuracy: {}'.format(wq_acc))

    fin_path = './sq_results/{}/'.format(args.arch)
    os.makedirs(fin_path, exist_ok=True)
    wq_path = fin_path + 'w{}b_lr{:.4f}_acc{:.4f}.pkl'.format(args.wbits, args.qps_lr, wq_acc)
    with open(wq_path, "wb") as fw:
        pickle.dump(qnn, fw)

