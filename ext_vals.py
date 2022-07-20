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

qps_pool = None
layer_id = 0
wbits = -1
arch = None
gpuid = None
qps_lr = None
