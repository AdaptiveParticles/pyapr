import sys, os
import math

#import pdb
import shutil

import numpy as np
from torch import nn
from torch.autograd import Function
from torch.nn import init
import torch

import _pyaprwrapper


def save_checkpoint(state, is_best, filedir, filename):
    torch.save(state, os.path.join(filedir, filename))
    if is_best:

        tmp = filename.split('.')
        fname = tmp[0] + '_best.pth.tar'
        shutil.copyfile(os.path.join(filedir, filename), os.path.join(filedir, fname))


# TODO: fix this guy
class APRInputLayer:
    def __init__(self, rgb=False, get_level=False, level_only=False):
        #super(APRInputLayer, self).__init__()

        self.rgb = rgb
        self.get_level = get_level
        self.level_only = level_only

    def __call__(self, aprList):
        batch_size = len(aprList)
        npartmax = 0
        for i in range(batch_size):
            npart = aprList[i].nparticles()
            npartmax = max(npartmax, npart)

        channels = 1 if self.level_only else (1 + (2 if self.rgb else 0) + (1 if self.get_level else 0))
        x = np.zeros((batch_size, channels, npartmax), dtype=np.float32)

        for i in range(batch_size):

            if self.get_level:
                lvl = aprList[i].get_levels()
                if npartmax > lvl.shape[0]:
                    x[i, -1, :] = np.concatenate((lvl, np.zeros(npartmax - len(lvl), dtype=np.float32)))
                else:
                    x[i, -1, :] = lvl

            if not self.level_only:
                tmp = aprList[i].get_intensities()

                if self.rgb:
                    tmp = tmp.reshape(3, -1)
                    if npartmax > tmp.shape[1]:
                        x[i, 0:3, :] = np.concatenate((tmp, np.zeros((3, npartmax-tmp.shape[1]), dtype=np.float32)), axis=1)
                    else:
                        x[i, 0:3, :] = tmp
                else:
                    if npartmax > tmp.shape[0]:
                        x[i, 0, :] = np.concatenate((tmp, np.zeros(npartmax - len(tmp), dtype=np.float32)))
                    else:
                        x[i, 0, :] = tmp

        level_deltas = torch.zeros(len(aprList), dtype=torch.int)

        return torch.from_numpy(x), level_deltas


class APRConvFunction(Function):
    @staticmethod
    def forward(ctx, intensities, weights, bias, aprs, level_deltas):

        ctx.apr = aprs

        dlevel = level_deltas.data.numpy()

        ctx.save_for_backward(intensities, weights, bias, torch.from_numpy(np.copy(dlevel)))

        output = np.zeros(shape=(intensities.shape[0], weights.shape[0], intensities.shape[2]), dtype=np.float32)

        dpo = _pyaprwrapper.DataParallelOps()

        dpo.convolve(aprs, intensities.data.numpy(), weights.data.numpy(), bias.data.numpy(), output, dlevel)

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        input_features, weights, bias, level_deltas = ctx.saved_tensors
        aprs = ctx.apr

        dlevel = level_deltas.data.numpy()

        d_input = np.zeros(input_features.shape, dtype=np.float32)
        d_weights = np.empty(weights.shape, dtype=np.float32)
        d_bias = np.empty(bias.shape, dtype=np.float32)

        dpo = _pyaprwrapper.DataParallelOps()

        dpo.convolve_backward(aprs, grad_output.data.numpy(), input_features.data.numpy(), weights.data.numpy(),
                              d_input, d_weights, d_bias, dlevel)

        return torch.from_numpy(d_input), torch.from_numpy(d_weights), torch.from_numpy(d_bias), None, None


class APRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nstencils):
        super(APRConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nstencils = nstencils

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, nstencils, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_features, apr, level_deltas):
        return APRConvFunction.apply(input_features, self.weight, self.bias, apr, level_deltas)


class APRConv3x3Function(Function):
    @staticmethod
    def forward(ctx, intensities, weights, bias, aprs, level_deltas):

        ctx.apr = aprs

        dlevel = level_deltas.data.numpy()

        ctx.save_for_backward(intensities, weights, bias, torch.from_numpy(np.copy(dlevel)))

        output = np.zeros(shape=(intensities.shape[0], weights.shape[0], intensities.shape[2]), dtype=np.float32)

        dpo = _pyaprwrapper.DataParallelOps()

        dpo.convolve3x3(aprs, intensities.data.numpy(), weights.data.numpy(), bias.data.numpy(), output, dlevel)

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        input_features, weights, bias, level_deltas = ctx.saved_tensors
        aprs = ctx.apr

        dlevel = level_deltas.data.numpy()

        d_input = np.zeros(input_features.shape, dtype=np.float32)
        d_weights = np.empty(weights.shape, dtype=np.float32)
        d_bias = np.empty(bias.shape, dtype=np.float32)

        dpo = _pyaprwrapper.DataParallelOps()

        dpo.convolve3x3_backward(aprs, grad_output.data.numpy(), input_features.data.numpy(), weights.data.numpy(),
                                 d_input, d_weights, d_bias, dlevel)

        return torch.from_numpy(d_input), torch.from_numpy(d_weights), torch.from_numpy(d_bias), None, None


class APRConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, nstencils):
        super(APRConv3x3, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nstencils = nstencils

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, nstencils, 3, 3))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_features, apr, level_deltas):
        return APRConv3x3Function.apply(input_features, self.weight, self.bias, apr, level_deltas)




class APRConv1x1Function(Function):
    @staticmethod
    def forward(ctx, intensities, weights, bias, aprs, level_deltas):

        ctx.apr = aprs

        dlevel = level_deltas.data.numpy()

        ctx.save_for_backward(intensities, weights, bias, torch.from_numpy(np.copy(dlevel)))

        output = np.zeros(shape=(intensities.shape[0], weights.shape[0], intensities.shape[2]), dtype=np.float32)

        dpo = _pyaprwrapper.DataParallelOps()

        dpo.convolve1x1(aprs, intensities.data.numpy(), weights.data.numpy(), bias.data.numpy(), output, dlevel)

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        input_features, weights, bias, level_deltas = ctx.saved_tensors
        aprs = ctx.apr

        dlevel = level_deltas.data.numpy()

        d_input = np.zeros(input_features.shape, dtype=np.float32)
        d_weights = np.empty(weights.shape, dtype=np.float32)
        d_bias = np.empty(bias.shape, dtype=np.float32)

        dpo = _pyaprwrapper.DataParallelOps()

        dpo.convolve1x1_backward(aprs, grad_output.data.numpy(), input_features.data.numpy(), weights.data.numpy(),
                                 d_input, d_weights, d_bias, dlevel)

        return torch.from_numpy(d_input), torch.from_numpy(d_weights), torch.from_numpy(d_bias), None, None


class APRConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, nstencils):
        super(APRConv1x1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nstencils = nstencils

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, nstencils, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_features, apr, level_deltas):
        return APRConv1x1Function.apply(input_features, self.weight, self.bias, apr, level_deltas)


class APRMaxPoolFunction(Function):
    @staticmethod
    def forward(ctx, intensities, apr, level_deltas):

        dlevel = level_deltas.data.numpy()

        ctx.apr = apr
        ctx.input_shape = intensities.shape

        npartmax = 0
        for i in range(len(apr)):
            npart = apr[i].number_particles_after_maxpool(dlevel[i])
            npartmax = max(npartmax, npart)

        output = -np.finfo(np.float32).max * np.ones(shape=(intensities.shape[0], intensities.shape[1], npartmax), dtype=np.float32)
        index_arr = -np.ones(output.shape, dtype=np.int64)

        dpo = _pyaprwrapper.DataParallelOps()
        dpo.max_pool(apr, intensities.data.numpy(), output, dlevel, index_arr)

        for i in range(level_deltas.shape[0]):
            level_deltas[i] += 1

        ctx.max_indices = index_arr

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        max_indices = ctx.max_indices
        grad_input = np.zeros(ctx.input_shape, dtype=np.float32)

        dpo = _pyaprwrapper.DataParallelOps()

        dpo.max_pool_backward(grad_output.data.numpy(), grad_input, max_indices)

        return torch.from_numpy(grad_input), None, None


class APRMaxPool(nn.Module):
    def __init__(self):
        super(APRMaxPool, self).__init__()

    def forward(self, input_features, apr, level_deltas):

        return APRMaxPoolFunction.apply(input_features, apr, level_deltas)

