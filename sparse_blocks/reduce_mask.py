import numpy as np
import torch
from torch.autograd import Function
from torch import nn

from sparse_blocks import _C
from sparse_blocks.utils import get_block_params


class ReduceMaskFunc(Function):
    @staticmethod
    def forward(ctx, mask, threshold, block_size, block_stride, block_offset, block_cnt, padding=None, avg_pool=False):
        ctx.N, C, ctx.H, ctx.W = mask.shape
        assert C == 1, "Mask should have a single channel"

        if mask.is_cuda:
            if len(mask.shape) >= 4:
                mask = mask.squeeze(1)  # mask should be (N, C=1, H, W)

            output = _C.reducemask_forward(mask,
                                           ctx.N, ctx.H, ctx.W, threshold,
                                           block_offset[0], block_offset[1],
                                           block_size[0], block_size[1],
                                           block_cnt[0], block_cnt[1],
                                           block_stride[0], block_stride[1],
                                           avg_pool)
        else:
            mask = nn.functional.pad(
                mask, padding, mode="constant")
            mask_ = nn.functional.max_pool2d(mask, block_size, block_stride, 0)
            mask_ = torch.squeeze(mask_, 1)  # remove channel dimension
            output = torch.nonzero(mask_ > threshold)

        # output = ReduceMaskFunc._sort(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None

    @staticmethod
    def _sort(x):
        # x is always (N,3)
        d = x.size(0)
        k = (d*d)*x[:, 0] + d*x[:, 1] + x[:, 2]
        _, idxs = torch.sort(k)
        return torch.index_select(x, 0, idxs)


class ReduceMask(nn.Module):
    def __init__(self, mask_size, threshold, block_size, kernel_size=(3, 3), stride=(1, 1), padding=None, avg_pool=False):
        super().__init__()
        self.threshold = threshold
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = avg_pool
        self.block_stride, \
            self.block_offset, \
            self.block_cnt, \
            self.padding = get_block_params(mask_size[0], mask_size[1],
                                            self.block_size, self.kernel_size,
                                            self.stride, padding=None)

    def forward(self, mask):
        y = ReduceMaskFunc.apply(mask, self.threshold,
                                 self.block_size, self.block_stride,
                                 self.block_offset, self.block_cnt,
                                 self.padding, self.avg_pool)
        return y, (self.block_size, self.block_stride, self.block_offset)
