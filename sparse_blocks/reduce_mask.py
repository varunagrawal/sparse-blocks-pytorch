import numpy as np
import torch
from torch.autograd import Function
from torch import nn

from sparse_blocks import _C
from sparse_blocks import utils


class ReduceMaskFunc(Function):
    @staticmethod
    def forward(ctx, mask, threshold, block_size, kernel_size=[3, 3], stride=[1, 1], padding="SAME", avg_pool=False):
        ctx.N, C, ctx.H, ctx.W = mask.shape
        assert C == 1, "Mask should have a single channel"
        pad_h0, pad_h1, pad_w0, pad_w1 = utils.get_padding(ctx.H, ctx.W,
                                                           kernel_size,
                                                           stride, padding)

        block_offset = [-pad_h0, -pad_w0]

        block_stride = [block_size[0] - kernel_size[0] + stride[0],
                        block_size[1] - kernel_size[1] + stride[1]]

        x_pad_shape = [ctx.H + pad_h0 + pad_h1,
                       ctx.W + pad_w0 + pad_w1]

        if padding == "SAME":
            out_shape = [int(np.ceil(float(x_pad_shape[0]) / stride[0])),
                         int(np.ceil(float(x_pad_shape[1]) / stride[1]))]
        elif padding == "VALID":
            out_shape = [int(np.ceil(float(x_pad_shape[0] - kernel_size[0] + 1) / stride[0])),
                         int(np.ceil(float(x_pad_shape[1] - kernel_size[1] + 1) / stride[1]))]

        block_cnt = [out_shape[0], out_shape[1]]

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
                mask, [pad_w0, pad_w1, pad_h0, pad_h1], mode="constant")
            mask_ = nn.functional.max_pool2d(mask, block_size, block_stride, 0)
            mask_ = torch.squeeze(mask_, 1)  # remove channel dimension
            output = torch.nonzero(mask_ > threshold)

        sorted_output = ReduceMaskFunc._sort(output)
        return sorted_output

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
    def __init__(self, threshold, block_size, kernel_size=[3, 3], stride=[1, 1], padding="SAME", avg_pool=False):
        super().__init__()
        self.threshold = threshold
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = avg_pool

    def forward(self, mask):
        return ReduceMaskFunc.apply(mask, self.threshold,
                                    self.block_size, self.kernel_size,
                                    self.stride, self.padding, self.avg_pool)
