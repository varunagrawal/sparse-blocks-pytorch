import numpy as np
import tensorflow as tf


def _calc_block_strides(bsize, ksize, strides):
    """Calculates strides for blocks.
    :param bsize:     [list]        List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:     [list]        List of 4 int. Sparse convolution kernel size.
    :param strides:   [list]        List of 4 int. Sparse convolution strides.
    :return           [list]        List of 4 int. Block strides.
    """
    return [1, bsize[1] - ksize[0] + strides[1], bsize[2] - ksize[1] + strides[2], 1]


def _get_offset_array(shape):
    """
    Computes the offset array used to upsample indices with NumPy (static).
    :param shape:   [list]     Window shape.
    """
    center = [int(ss - 1) // 2 for ss in shape]
    axes = [np.arange(-cc, int(ss) - cc).astype(np.int32)
            for cc, ss in zip(center, shape)]
    if len(shape) > 1:
        for jj in range(len(shape)):
            for ii in range(len(shape) + 1):
                if ii != jj:
                    axes[jj] = np.expand_dims(axes[jj], ii)
        for jj in range(len(shape)):
            shape_ = [int(ss) for ss in shape] + [1]
            shape_[jj] = 1
            axes[jj] = np.tile(axes[jj], shape_)
        offset = np.concatenate(axes, len(shape))
        return tf.constant(offset)
    else:
        return tf.constant(axes[0])


def _check_strides(strides):
    """
    Validates strides parameters.
    :param strides:  [list]    List of 4 int or a Tensor of 4 elements. Convolution stride size.
    :returns:        [list]    List of 4 int or a Tensor of 4 elements, if inputs are valid.
    """
    if type(strides) == list or type(strides) == tuple:
        assert len(strides) == 4, 'Expect `strides` a list/tuple of length 4.'
        assert strides[0] == strides[3] == 1, 'Expect first and last dimension of `strides` = 1.'
    elif type(strides) == tf.Tensor:
        assert len(strides.get_shape()
                   ) == 1, 'Expect `strides` a rank 1 Tensor.'
        assert int(strides.get_shape()[
                   0]) == 4, 'Expect `strides` to have 4 elements.'
        assert_strides = tf.assert_equal(
            tf.stack([strides[0], strides[3]]),
            tf.constant([1, 1], dtype=strides.dtype),
            message='Expect first and last dimension of `strides` = 1.')
        with tf.control_dependencies([assert_strides]):
            strides = tf.cast(strides, tf.int32)
    else:
        assert False, '`strides` has unknown type: {}'.format(type(strides))
    return strides


def _check_ksize(ksize):
    """
    Validates ksize parameters.
    :param ksize:    [list]    List of 4 int or a Tensor of 4 elements. Convolution kernel size.
    :returns:        [list]    List of 4 int or a Tensor of 4 elements, if inputs are valid.
    """
    if type(ksize) == list or type(ksize) == tuple:
        assert len(ksize) == 4, 'Expect `ksize` a list/tuple of length 4.'
    elif type(ksize) == tf.Tensor:
        assert len(ksize.get_shape()) == 1, 'Expect `ksize` a rank 1 Tensor.'
        assert int(ksize.get_shape()[
                   0]) == 4, 'Expect `ksize` to have 4 elements.'
        ksize = tf.cast(ksize, tf.int32)
    else:
        assert False, '`ksize` has unknown type: {}'.format(type(ksize))
    return ksize


def _div_padding(pad_size):
    """
    Divides padding to two sides so that the features are centered.
    :param pad_size: [Tensor]  Scalar. Padding size.
    :return          [Tensor]  Scalar. First padding size.
    :return          [Tensor]  Scalar. Second padding size.
    """
    return tf.cast(tf.floor(tf.to_float(pad_size) / 2.0), tf.int32), tf.cast(
        tf.ceil(tf.to_float(pad_size) / 2.0), tf.int32)


def _div_padding_np(pad_size):
    """
    Divides padding to two sides so that the features are centered.
    :param pad_size: [np.ndarray]  Scalar. Padding size.
    :return          [int]  Scalar. First padding size.
    :return          [int]  Scalar. Second padding size.
    """
    return int(np.floor(float(pad_size) / 2.0)), int(np.ceil(float(pad_size) / 2.0))


def calc_out_size_1d(in_size, ksize, stride, padding):
    """
    Calculates output size on one dimension.
    :param in_size:  [int]     Input size.
    :param ksize:    [int]     Kernel size.
    :param stride:   [int]     Stride size.
    :param pad:      [string]  Padding method, `SAME` or `VALID`.
    :return          [int]     Output size.
    """

    if padding == 'VALID':
        return tf.cast(tf.ceil(tf.to_float(in_size - ksize + 1) / tf.to_float(stride)), tf.int32)
    elif padding == 'SAME':
        return tf.cast(tf.ceil(tf.to_float(in_size) / tf.to_float(stride)), tf.int32)
    else:
        raise ValueError('Unknown padding method \"{}\"'.format(padding))


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


def calc_padding_4d(in_shape, ksize, strides, padding):
    """
    Calculates padding width on four dimensions: top, bottom, left, and right.
    :param x:        [Tensor]  Input tensor.
    :param ksize     [list]    List of 4 int or a Tensor of 4 elements. Convolution kernel size.
    :param strides   [list]    List of 4 int or a Tensor of 4 elements. Convolution stride size.
    :param padding   [list]    Padding method, `VALID` or `SAME`.
    :return          [tuple]   Tuple of 4 int. Padding length on top, bottom, left, and right.
    """
    ksize = _check_ksize(ksize)
    strides = _check_strides(strides)
    if padding == 'VALID':
        return 0, 0, 0, 0
    elif padding == 'SAME':
        if type(in_shape[1]) == int:
            out_size_h = calc_out_size_1d_np(
                in_shape[1], ksize[0], strides[1], padding)
            out_size_w = calc_out_size_1d_np(
                in_shape[2], ksize[1], strides[2], padding)
        elif type(in_shape[1]) == tf.Tensor:
            out_size_h = calc_out_size_1d(
                in_shape[1], ksize[0], strides[1], padding)
            out_size_w = calc_out_size_1d(
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
        elif type(pad_h) == tf.Tensor:
            pad_h0, pad_h1 = _div_padding(pad_h)
            pad_w0, pad_w1 = _div_padding(pad_w)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(pad_h)))
        return pad_h0, pad_h1, pad_w0, pad_w1
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
        elif type(_pad) == tf.Tensor:
            return tf.maximum(_pad, 0)
        else:
            raise ValueError('Unknown type \"{}\"'.format(type(_pad)))
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
    x_shape = tf.shape(x)
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
            pad_h1 += tf.mod(-h + bsize[1], bstrides[1])
            pad_w1 += tf.mod(-w + bsize[2], bstrides[2])
        return tf.pad(x, [[0, 0], [pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]])
    else:
        if bstrides is not None:
            assert bsize is not None, 'Must pass in bsize and bstrides together.'
            h = x_shape[1]
            w = x_shape[2]
            pad_h1 = tf.mod(-h + bsize[1], bstrides[1])
            pad_w1 = tf.mod(-w + bsize[2], bstrides[2])
            return tf.cond(
                tf.logical_or(tf.greater(pad_h1, 0), tf.greater(pad_w1, 0)),
                lambda: tf.pad(x, [[0, 0], [0, pad_h1], [0, pad_w1], [0, 0]]), lambda: x)
        else:
            return x


def upsample_indices(indices, ksize, strides):
    """
    Upsamples the indices to have all indices in a rectangle.
    :param indices:   [Tensor]      [M, 3]. Center locations (N, H, W) of the M rectangles.
                                    Dtype int32.
    :param ksize:     [list]        Size of the rectangle, or downsample ratio.
    :param strides:   [list]        Strides of the pooling operation.
    :return           [Tensor]      [M, h, w, 3]. Locations of all pixels in the rectangles.
                                    Dtype int32.
    """
    assert len(indices.get_shape()) == 2, 'Expect indices rank = 2'
    assert ksize[0] == ksize[3] == 1, 'Expect first and last dimensions of ksize = 1'
    assert strides[0] == strides[3] == 1, 'Expect first and last dimensions of strides = 1, {}'.format(
        strides)
    h_scale = strides[1]
    w_scale = strides[2]
    scale = tf.stack([1, h_scale, w_scale])
    indices *= scale
    # Since we always use VALID to perform pooling, shift is needed here.
    shift = tf.stack([0, (ksize[1] - 1) // 2, (ksize[2] - 1) // 2])
    indices += shift
    indices_ = tf.expand_dims(tf.expand_dims(indices, 1), 2)
    # indices_ = tf.tile(indices_, [1, ksize[1], ksize[2], 1])
    offset = _get_offset_array(ksize[0:3])
    indices_ += offset
    return indices_


def convert_mask_to_indices(mask, bsize, ksize, strides, padding, tol):
    """
    Converts a binary mask to sparse indices.
    :param mask:     [Tensor]   [N, H, W]. 1 indicates non-sparse locations. Dtype float32.
    :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
                                Currently only supports when,
                                1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                                2) (bsize[2] - ksize[1]) % strides[2] == 0
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
    :param tol:      [float]    Lower bound of occupancy for creating a rectangle.
    :return          [Tensor]   [M, 3]. Center locations (N, H, W) of M rectangles. Dtype int32.
    """
    ERR_MSG_RANK = 'Expect mask rank = 3'
    ERR_MSG_DIV = 'Expect `stride` divides `bsize` - `ksize`. stride {}, bsize {}, ksize {}.'
    ERR_MSG_DIM = 'Expect first and last dimensions of strides = 1. Dim {}.'

    assert len(mask.get_shape()) == 3, ERR_MSG_RANK
    assert type(bsize) in [list, tuple], '`bsize` needs to be a list or tuple.'
    assert type(ksize) in [list, tuple], '`ksize` needs to be a list or tuple.'
    assert type(strides) in [
        list, tuple], '`strides` needs to be a list or tuple.'
    assert (bsize[1] - ksize[0]) % strides[1] == 0, ERR_MSG_DIV.format(
        strides[1], bsize[1], ksize[0])
    assert (bsize[2] - ksize[1]) % strides[2] == 0, ERR_MSG_DIV.format(
        strides[2], bsize[2], ksize[1])
    assert strides[0] == strides[3] == 1, ERR_MSG_DIM.format(strides)

    bstrides = _calc_block_strides(bsize, ksize, strides)

    # Pad mask.
    mask_ = tf.expand_dims(mask, 3)
    mask_ = _pad_input(mask_, ksize, strides, padding,
                       bsize=bsize, bstrides=bstrides)
    # Blocks are always valid conv.
    mask_ = tf.nn.max_pool(mask_, bsize, bstrides, 'VALID')
    mask_ = tf.squeeze(mask_, [3])
    indices = tf.where(tf.greater(mask_, tol))
    indices = tf.cast(indices, tf.int32)
    return indices


def convert_mask_to_block_indices(mask, bsize, ksize, strides, padding, tol):
    """
    Converts a binary mask to block sparse indices.
    :param mask:     [Tensor]   [N, H, W]. 1 indicates non-sparse locations. Dtype float32.
    :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
                                Currently only supports when,
                                1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                                2) (bsize[2] - ksize[1]) % strides[2] == 0
    :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
    :param tol:      [float]    Lower bound of occupancy for creating a rectangle.
    :return          [Tensor]   [M, h, w, 3]. Pixel locations of M rectangles. Dtype int32.
    """
    indices = convert_mask_to_indices(
        mask, bsize, ksize, strides, padding, tol)
    bstrides = _calc_block_strides(bsize, ksize, strides)
    blk_indices = upsample_indices(indices, bsize, bstrides)
    return blk_indices
