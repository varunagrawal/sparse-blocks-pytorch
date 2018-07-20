import numpy as np


def get_padding(height, width, kernel_size=(3, 3), strides=(1, 1), padding=None):
    """

    Parameters
    ----------
    in_shape :
        Tuple of mask shape (N,H,W)
    kernel_size : list, optional
        Kernel size of pooling layer (the default is [3, 3])
    strides : list, optional
        (the default is [1, 1], which [default_description])
    padding : [type], optional
        (the default is None, which [default_description])

    Returns
    -------
    Padding sizes
    """
    if padding is None or padding == "VALID":
        return 0, 0, 0, 0

    elif padding == "SAME":  # SAME padding means the output is the same size as the input
        out_h = int(np.ceil(float(height) / float(strides[0])))
        out_w = int(np.ceil(float(width) / float(strides[1])))

        pad_h = max(((out_h - 1)*strides[0] + kernel_size[0] - height), 0)
        pad_w = max(((out_w - 1)*strides[1] + kernel_size[1] - width), 0)

        pad_h0, pad_h1 = int(np.floor(float(pad_h) / 2.0)), \
            int(np.ceil(float(pad_h) / 2.0))

        pad_w0, pad_w1 = int(np.floor(float(pad_w) / 2.0)), \
            int(np.ceil(float(pad_w) / 2.0))

        return pad_h0, pad_h1, pad_w0, pad_w1


def get_out_size_1d(in_size, ksize, stride, padding):
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


def get_output_shape(input_shape, out_channels, kernel_size, strides, padding):
    """Calculates output shape (rank 4) of a 2D convolution operation.

    :param input_shape:  [list]    Input tensor shape.
    :param out_channels: [list]    Number of channels in output.
    :param kernel_size:  [list]    Kernel shape.
    :param strides:      [list]    Strides list.
    :param padding:      [string]  Padding method, `SAME` or `VALID`.

    :return              [list]    Output tensor shape.
    """
    assert len(input_shape) == 4, "input_shape should be of length 4 (NCHW)"
    return [
        input_shape[0],
        out_channels,
        get_out_size_1d(input_shape[2], kernel_size[0], strides[0], padding),
        get_out_size_1d(input_shape[3], kernel_size[1], strides[1], padding),
    ]


def get_block_params(H, W, block_size, kernel_size=(3, 3), stride=(1, 1), padding=None):
    if padding is None:
        padding = "SAME"

    pad_h0, pad_h1, pad_w0, pad_w1 = get_padding(H, W,
                                                 kernel_size,
                                                 stride, padding)

    block_offset = [-pad_h0, -pad_w0]

    block_stride = [block_size[0] - kernel_size[0] + stride[0],
                    block_size[1] - kernel_size[1] + stride[1]]

    x_pad_shape = [H + pad_h0 + pad_h1,
                   W + pad_w0 + pad_w1]

    out_shape = [get_out_size_1d(x_pad_shape[0], kernel_size[0], stride[0], padding),
                 get_out_size_1d(x_pad_shape[1], kernel_size[1], stride[1], padding)]

    block_cnt = [out_shape[0], out_shape[1]]

    return block_stride, block_offset, block_cnt, (pad_w0, pad_w1, pad_h0, pad_h1)
