import sys, os
import math

#import pdb
import shutil

import numpy as np
from torch import nn
from torch.autograd import Function
from torch.nn import init
import torch

import _pyaprwrapper.nn as aprnn


class APRInputLayer:
    def __call__(self, apr_arr, parts_arr, dtype=np.float32):

        batch_size = len(apr_arr)
        #assert parts_arr.shape[0] == batch_size

        #if parts_arr.ndim == 1:
        nch = 1
        #parts_arr = np.expand_dims(parts_arr, -1)
        #elif parts_arr.ndim == 2:
        #    nch = parts_arr.shape[1]
        #else:
        #    raise AssertionError('array of input particles must be of dimension 1 or 2')

        npartmax = 0
        for apr in apr_arr:
            npart = apr.total_number_particles()
            npartmax = max(npart, npartmax)

        x = np.empty((batch_size, nch, npartmax), dtype=dtype)

        for i in range(len(parts_arr)):
            #for j in range(nch):
            tmp = np.array(parts_arr[i], copy=False)
            npart = len(tmp)

            x[i, 0, :npart] = tmp
            x[i, 0, npart:] = 0

        dlvl = torch.zeros(batch_size, dtype=torch.int)

        return torch.from_numpy(x), dlvl


class APRConvFunction(Function):
    @staticmethod
    def forward(ctx, intensities, weights, bias, aprs, level_deltas):

        ctx.apr = aprs

        dlevel = level_deltas.data.numpy()

        ctx.save_for_backward(intensities, weights, bias, torch.from_numpy(np.copy(dlevel)))

        output = np.zeros(shape=(intensities.shape[0], weights.shape[0], intensities.shape[2]), dtype=intensities.data.numpy().dtype)

        aprnn.convolve(aprs, intensities.data.numpy(), weights.data.numpy(), bias.data.numpy(), output, dlevel)

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        input_features, weights, bias, level_deltas = ctx.saved_tensors
        aprs = ctx.apr

        dlevel = level_deltas.data.numpy()

        d_input = np.zeros(input_features.shape, dtype=input_features.data.numpy().dtype)
        d_weights = np.empty(weights.shape, dtype=np.float32)
        d_bias = np.empty(bias.shape, dtype=np.float32)

        aprnn.convolve_backward(aprs, grad_output.data.numpy(), input_features.data.numpy(), weights.data.numpy(),
                                d_input, d_weights, d_bias, dlevel)

        return torch.from_numpy(d_input), torch.from_numpy(d_weights), torch.from_numpy(d_bias), None, None


class APRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nstencils):
        super(APRConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
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

        if self.kernel_size == 1:

            return APRConv1x1Function.apply(input_features, self.weight, self.bias, apr, level_deltas)

        elif self.kernel_size == 3:

            return APRConv3x3Function.apply(input_features, self.weight, self.bias, apr, level_deltas)

        else:

            return APRConvFunction.apply(input_features, self.weight, self.bias, apr, level_deltas)


class APRConv3x3Function(Function):
    @staticmethod
    def forward(ctx, intensities, weights, bias, aprs, level_deltas):

        ctx.apr = aprs

        dlevel = level_deltas.data.numpy()

        ctx.save_for_backward(intensities, weights, bias, torch.from_numpy(np.copy(dlevel)))

        output = np.zeros(shape=(intensities.shape[0], weights.shape[0], intensities.shape[2]), dtype=intensities.data.numpy().dtype)

        aprnn.convolve3x3(aprs, intensities.data.numpy(), weights.data.numpy(), bias.data.numpy(), output, dlevel)

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        input_features, weights, bias, level_deltas = ctx.saved_tensors
        aprs = ctx.apr

        dlevel = level_deltas.data.numpy()

        d_input = np.zeros(input_features.shape, dtype=input_features.data.numpy().dtype)
        d_weights = np.empty(weights.shape, dtype=np.float32)
        d_bias = np.empty(bias.shape, dtype=np.float32)

        aprnn.convolve3x3_backward(aprs, grad_output.data.numpy(), input_features.data.numpy(), weights.data.numpy(),
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

        output = np.zeros(shape=(intensities.shape[0], weights.shape[0], intensities.shape[2]), dtype=intensities.data.numpy().dtype)

        aprnn.convolve1x1(aprs, intensities.data.numpy(), weights.data.numpy(), bias.data.numpy(), output, dlevel)

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        input_features, weights, bias, level_deltas = ctx.saved_tensors
        aprs = ctx.apr

        dlevel = level_deltas.data.numpy()
        np_input = input_features.data.numpy()

        d_input = np.zeros(input_features.shape, dtype=np_input.dtype)
        d_weights = np.empty(weights.shape, dtype=np.float32)
        d_bias = np.empty(bias.shape, dtype=np.float32)

        aprnn.convolve1x1_backward(aprs, grad_output.data.numpy(), np_input, weights.data.numpy(),
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
    def forward(ctx, intensities, apr, level_deltas, inc_dlvl):

        dlevel = level_deltas.data.numpy()

        ctx.apr = apr
        ctx.input_shape = intensities.shape

        npartmax = 0
        for i in range(len(apr)):
            npart = aprnn.number_particles_after_pool(apr[i], dlevel[i])
            npartmax = max(npartmax, npart)

        output = -(np.finfo(np.float32).max / 2) * np.ones(shape=(intensities.shape[0], intensities.shape[1], npartmax), dtype=intensities.data.numpy().dtype)
        index_arr = -np.ones(output.shape, dtype=np.int64)

        aprnn.max_pool(apr, intensities.data.numpy(), output, dlevel, index_arr)

        if inc_dlvl:
            for i in range(level_deltas.shape[0]):
                level_deltas[i] += 1

        ctx.max_indices = index_arr

        return torch.from_numpy(output)

    @staticmethod
    def backward(ctx, grad_output):

        max_indices = ctx.max_indices
        grad_input = np.zeros(ctx.input_shape, dtype=grad_output.data.numpy().dtype)

        aprnn.max_pool_backward(grad_output.data.numpy(), grad_input, max_indices)

        return torch.from_numpy(grad_input), None, None, None


class APRMaxPool(nn.Module):
    def __init__(self, increment_level_delta=True):
        super(APRMaxPool, self).__init__()

        self.increment_level_delta=increment_level_delta

    def forward(self, input_features, apr, level_deltas):

        return APRMaxPoolFunction.apply(input_features, apr, level_deltas, self.increment_level_delta)
