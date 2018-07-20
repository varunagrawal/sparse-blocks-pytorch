import torch
from sparse_blocks import ReduceMask
import numpy as np
from torch import nn


def calc_out_size_1d_np(in_size, ksize, stride, padding):
    """
    Calculates output size on one dimension.
    :param in_size:  [int]     Input size.
    :param ksize:    [int]     Kernel size.
    :param stride:   [int]     Stride size.
    :param pad:      [string]  Padding method, `SAME` or `VALID`.
    :return          [int]     Output size.
    """

    if padding == 'VALID':
        return int(np.ceil(float(in_size - ksize + 1) / float(stride)))
    elif padding == 'SAME':
        return int(np.ceil(float(in_size) / float(stride)))
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


def calc_padding_1d(in_size, out_size, ksize, stride, padding):
    """
    Calculates padding width on one dimension.
    :param in_size:  [Tensor]  Scalar. Input size.
    :param out_size: [Tensor]  Scalar. Output size.
    :param ksize:    [Tensor]  Scalar or int. Kernel size.
    :param strides:  [Tensor]  Scalar or int. Stride size.
    :param padding:  [string]  Padding method, `SAME` or `VALID`.
    :returns:        [Tensor]  Scalar. Padding size.
    """
    if padding == 'VALID':
        return 0
    elif padding == 'SAME':
        _pad = (out_size - 1) * stride + ksize - in_size
        if type(_pad) == int:
            return max(_pad, 0)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(_pad)))
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


def _div_padding_np(pad_size):
    """
    Divides padding to two sides so that the features are centered.
    :param pad_size: [np.ndarray]  Scalar. Padding size.
    :return          [int]  Scalar. First padding size.
    :return          [int]  Scalar. Second padding size.
    """
    return int(np.floor(float(pad_size) / 2.0)), int(np.ceil(float(pad_size) / 2.0))


def calc_padding_4d(in_shape, ksize, strides, padding):
    """
    Calculates padding width on four dimensions: top, bottom, left, and right.
    :param x:        [Tensor]  Input tensor.
    :param ksize     [list]    List of 4 int or a Tensor of 4 elements. Convolution kernel size.
    :param strides   [list]    List of 4 int or a Tensor of 4 elements. Convolution stride size.
    :param padding   [list]    Padding method, `VALID` or `SAME`.
    :return          [tuple]   Tuple of 4 int. Padding length on top, bottom, left, and right.
    """
    if padding == 'VALID':
        return 0, 0, 0, 0
    elif padding == 'SAME':
        if type(in_shape[1]) == int:
            out_size_h = calc_out_size_1d_np(
                in_shape[1], ksize[0], strides[1], padding)
            out_size_w = calc_out_size_1d_np(
                in_shape[2], ksize[1], strides[2], padding)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(in_shape[1])))
        pad_h = calc_padding_1d(
            in_shape[1], out_size_h, ksize[0], strides[1], padding)
        pad_w = calc_padding_1d(
            in_shape[2], out_size_w, ksize[1], strides[2], padding)
        if type(pad_h) == int:
            pad_h0, pad_h1 = _div_padding_np(pad_h)
            pad_w0, pad_w1 = _div_padding_np(pad_w)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(pad_h)))
        return pad_h0, pad_h1, pad_w0, pad_w1
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


def _pad_input(x, ksize, strides, padding, bsize=None, bstrides=None):
    """Pads the input tensor.
    Optional to pass in block strides. The right hand side padding will be increased if the last
    block does not fit in (no effect on the convolution results.
    :param x:        [Tensor]   [N, H, W, C]. input tensor, dtype float32.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
    :param bsize     [list]     List of 4 int. Block size. Optional.
    :param bstrides: [list]     List of 4 int. Block strides. Optional.
    :return          [Tensor]   [N, H+Ph, W+Pw, C]. Padded input tensor.
    """
    x_shape = x.shape
    if padding == 'SAME':
        pad_h0, pad_h1, pad_w0, pad_w1 = calc_padding_4d(
            x_shape, ksize, strides, padding)

        if bstrides is not None:
            # Here we do not use the standard padding on the right hand side.
            # If the convolution results is larger than expected, the scatter function will not use
            # out-of-boundary points.
            assert bsize is not None, 'Must pass in bsize and bstrides together.'
            h = x_shape[1] + pad_h0 + pad_h1
            w = x_shape[2] + pad_w0 + pad_w1
            pad_h1 += np.mod(-h + bsize[1], bstrides[1])
            pad_w1 += np.mod(-w + bsize[2], bstrides[2])
        return np.pad(x, [[0, 0], [pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]], mode='constant')
    else:
        if bstrides is not None:
            assert bsize is not None, 'Must pass in bsize and bstrides together.'
            h = x_shape[1]
            w = x_shape[2]
            pad_h1 = np.mod(-h + bsize[1], bstrides[1])
            pad_w1 = np.mod(-w + bsize[2], bstrides[2])
            if np.all(np.logical_or(np.greater(pad_h1, 0), np.greater(pad_w1, 0))):
                return lambda: np.pad(x, [[0, 0], [0, pad_h1], [0, pad_w1], [0, 0]], mode='constant')
            else:
                return lambda: x

        else:
            return x


def _calc_block_strides(bsize, ksize, strides):
    """Calculates strides for blocks.
    :param bsize:     [list]        List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:     [list]        List of 4 int. Sparse convolution kernel size.
    :param strides:   [list]        List of 4 int. Sparse convolution strides.
    :return           [list]        List of 4 int. Block strides.
    """
    return [1, bsize[1] - ksize[0] + strides[1], bsize[2] - ksize[1] + strides[2], 1]


def convert_mask_to_indices(mask, block_size, kernel_size, stride, padding, threshold):
    bstrides = _calc_block_strides(block_size, kernel_size, stride)
    # print(mask[0, :, :])
    mask = mask[:, :, :, np.newaxis]
    mask_ = _pad_input(mask, kernel_size, stride, padding,
                       bsize=block_size, bstrides=bstrides)

    # print("Mask padding", mask_.shape)
    # print(mask_[0, :, :, 0])
    # Blocks are always valid conv.
    mask = torch.from_numpy(mask_.transpose((0, 3, 1, 2)))
    mask_ = nn.functional.max_pool2d(mask, block_size[1:3], bstrides[0:2], 0)
    mask_ = torch.squeeze(mask_)  # .numpy()
    # print(mask_)
    indices = torch.nonzero(mask_ > threshold)
    return indices


def test_reduce_mask():
    mask = np.array(
        [[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]],
        dtype=np.float32)

    gt_indices = convert_mask_to_indices(mask, block_size=[1, 3, 3, 1],
                                         kernel_size=[3, 3, 1, 1],
                                         stride=[1, 1, 1], padding="SAME", threshold=0)
    # print("gt_indices\n", gt_indices)
    mask = torch.from_numpy(mask)
    mask.unsqueeze_(1)

    red = ReduceMask(0, block_size=[3, 3], kernel_size=[3, 3])
    y, _ = red(mask)
    y = y[:, 1:3]
    # print("CPU", y, y.shape)

    assert np.array_equal(y.numpy(), gt_indices), "ReduceMask CPU is incorrect"

    red = red.cuda()
    y, _ = red(mask.cuda())
    # print("GPU", y, y.shape)
    y = y[:, 1:3].cpu()
    assert np.array_equal(y.numpy(), gt_indices), "ReduceMask GPU is incorrect"
