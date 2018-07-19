import numpy as np
import torch
from torch.autograd import Function
from torch import nn

from sparse_blocks import _C
from sparse_blocks import utils


class SparseGatherFunc(Function):
    @staticmethod
    def forward(ctx, x, indices, block_size, block_stride, block_offset):
        ctx.block_size = block_size
        ctx.block_stride = block_stride
        ctx.block_offset = block_offset

        # The extension expects the input tensor as (NHWC)
        x = x.permute(0, 2, 3, 1)

        ctx.save_for_backward(x, indices)

        y = _C.sparse_gather_forward(x, indices,
                                     block_size[0], block_size[1],
                                     block_stride[0], block_stride[1],
                                     block_offset[0], block_offset[1])

        # output is (NCWH)
        return y

    @staticmethod
    def backward(ctx, dy):
        _, indices = ctx.saved_tensors
        # output base tensor to add on top of. ctx.x should be (NHWC)
        base = torch.zeros(ctx.x.size())

        # The extension expects the input tensor as (NHWC)
        dy = dy.permute(0, 2, 3, 1)

        dx = _C.sparse_scatter_forward(dy, indices, base,
                                       ctx.block_size[0], ctx.block_size[1],
                                       ctx.block_stride[0], ctx.block_stride[1],
                                       ctx.block_offset[0], ctx.block_offset[1],
                                       True, True)

        # output is (NCWH)
        return dx


class SparseScatterFunc(Function):
    @staticmethod
    def forward(ctx, x, y_base, indices, block_size, block_stride, block_offset, add=False, atomic=False):
        ctx.block_size = block_size
        ctx.block_stride = block_stride
        ctx.block_offset = block_offset
        ctx.add = add
        ctx.atomic = atomic

        # The extension expects the input tensor as (NHWC)
        x = x.permute(0, 2, 3, 1)
        # y_base = y_base.permute(0, 2, 3, 1)

        ctx.save_for_backward(x, indices)

        y = _C.sparse_scatter_forward(x, indices, y_base,
                                      block_size[0], block_size[1],
                                      block_stride[0], block_stride[1],
                                      block_offset[0], block_offset[1],
                                      add, atomic)

        # output is (NCWH)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, indices = ctx.saved_tensors

        dx = _C.sparse_gather_forward(dy, indices,
                                      ctx.block_size[0], ctx.block_size[1],
                                      ctx.block_stride[0], ctx.block_stride[1],
                                      ctx.block_offset[0], ctx.block_offset[1])

        # return a list of gradients of output with respect to each input
        if not ctx.add:
            dy_reshaped = dy.permute(0, 2, 3, 1)
            dy_base = torch.ones(dy_reshaped.shape)
            # scatter blocks of zeroes over a base tensor of ones to compute a stamp-out gradient mask for dy_dybase

            stamp_out_blocks = _C.sparse_scatter_forward(torch.zeros(x.size()), indices, dy_base,
                                                         ctx.block_size[0], ctx.block_size[1],
                                                         ctx.block_stride[0], ctx.block_stride[1],
                                                         ctx.block_size[0], ctx.block_offset[1],
                                                         False, ctx.atomic)
            dy_dybase = dy * stamp_out_blocks

            return dx, dy_dybase

        else:
            # d(x+ybase)/dybase = 1, so just pass back grad as dout_dybase
            return dx, dy


class SparseGather(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices, block_size, block_stride, block_offset):
        return SparseGatherFunc.apply(x, indices,
                                      block_size, block_stride, block_offset)


class SparseScatter(nn.Module):
    def __init__(self, add=False, atomic=False):
        super().__init__()
        self.add = add
        self.atomic = atomic

    def forward(self, x, y_base, indices, block_size, block_stride, block_offset):
        return SparseScatterFunc.apply(x, y_base, indices,
                                       block_size, block_stride,
                                       block_offset, self.add, self.atomic)
