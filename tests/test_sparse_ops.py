import numpy as np
import torch
from torch import nn
from sparse_blocks import SparseGather, SparseScatter, ReduceMask

from utils import convert_mask_to_block_indices
import tensorflow as tf

from fixtures import gt_dict


def test_sparse_gather(channels=1):
    block_size = [3, 3]
    kernel_size = [3, 3]
    stride = [1, 1]
    mask = np.array(
        [[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]],
        dtype=np.float32)

    # x = np.arange(mask.shape[1] * mask.shape[2]).reshape([
    #     1, 1, mask.shape[1], mask.shape[2]]).astype(np.float32)
    x = np.arange(mask.shape[1] * mask.shape[2] * channels).reshape([1, mask.shape[1], mask.shape[2],
                                                                     channels]).astype(np.float32)
    x = torch.from_numpy(x.transpose(0, 3, 1, 2)).cuda()
    # print("Input", x, x.shape)

    mask = torch.from_numpy(mask[np.newaxis, :, :, :]).cuda()

    reduc = ReduceMask(0, block_size, kernel_size, stride,
                       padding="SAME", avg_pool=False)
    gat = SparseGather()

    indices, (block_size, block_stride, block_offset) = reduc(mask)

    gat_feat = gat(x, indices.int(), block_size, block_stride, block_offset)
    # print("Sparse Gather results\n", gat_feat, gat_feat.shape)

    gt = gt_dict['gather_c'+str(channels)]

    # print("Ground truth", gt, gt.shape)
    p = gat_feat.cpu().numpy()  # .transpose(0, 2, 3, 1)
    # print(p.shape)

    assert np.array_equal(gt, p), "Error Sparse Gather"
    assert np.array_equal(gt.shape, p.shape)
    if np.array_equal(gt, p):
        print("Sparse Gather is correct")


def test_sparse_scatter(channels=1):
    block_size = [3, 3]
    kernel_size = [3, 3]
    stride = [1, 1]
    out_shape = [1, channels, 5, 5]
    w = torch.ones([channels, channels, 3, 3]).cuda()

    mask = np.array(
        [[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]],
        dtype=np.float32)

    # x = np.arange(mask.shape[1] * mask.shape[2]).reshape([
    #     1, 1, mask.shape[1], mask.shape[2]]).astype(np.float32)
    x = np.arange(mask.shape[1] * mask.shape[2] * channels).reshape([1, mask.shape[1], mask.shape[2],
                                                                     channels]).astype(np.float32)
    x = torch.from_numpy(x.transpose(0, 3, 1, 2)).cuda()
    mask = torch.from_numpy(mask[np.newaxis, :, :, :]).cuda()

    reduc = ReduceMask(0, block_size, kernel_size, stride,
                       padding="SAME", avg_pool=False)
    gat = SparseGather()
    scat = SparseScatter(out_channels=channels,
                         kernel_size=kernel_size,
                         stride=stride)

    indices, (block_size, block_stride, block_offset) = reduc(mask)
    # print("indices", indices)

    p = gat(x, indices, block_size, block_stride, block_offset)
    # print("gather results", p)

    q = nn.functional.conv2d(p, w, stride=stride)

    conv_gt = gt_dict["conv_c"+str(channels)]
    # print(q, q.shape)

    assert np.array_equal(conv_gt, q.cpu().numpy()), \
        "Convolution results are incorrect"
    # print("Conv results", q, q.shape)

    # x, y_base, indices, block_size, block_stride, block_offset
    y_base = torch.zeros(out_shape).cuda()

    y = scat(q, y_base, indices, block_size, block_stride)
    # print(scat_feat, scat_feat.shape)

    scatter_gt = gt_dict["scatter_c"+str(channels)]

    # print("Scatter results", y, y.shape)
    assert np.array_equal(scatter_gt, y.cpu().numpy())
    if np.array_equal(scatter_gt, y.cpu().numpy()):
        print("Sparse Scatter is correct!")


if __name__ == "__main__":
    for c in [1, 2]:
        print("Testing with {} channels".format(c))
        test_sparse_gather(c)
        test_sparse_scatter(c)
