import numpy as np
import torch
from torch import nn
from sparse_blocks import SparseGather, SparseScatter, ReduceMask

from fixtures import gt_dict


def sort(x):
    # x is always (N,3)
    d = x.size(0)
    k = (d*d)*x[:, 0] + d*x[:, 1] + x[:, 2]
    _, idxs = torch.sort(k)
    return torch.index_select(x, 0, idxs)


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

    reduc = ReduceMask(mask.shape[2:4], 0, block_size, kernel_size, stride,
                       padding="SAME", avg_pool=False)
    gat = SparseGather()

    indices, (block_size, block_stride, block_offset) = reduc(mask)
    # print(indices)
    indices = sort(indices)

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


def test_sparse_scatter(channels=2):
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

    reduc = ReduceMask(mask.shape[2:4], 0, block_size, kernel_size, stride,
                       padding="SAME", avg_pool=False)
    gat = SparseGather()
    scat = SparseScatter(out_channels=channels,
                         kernel_size=kernel_size,
                         stride=stride, padding="VALID")

    indices, (block_size, block_stride, block_offset) = reduc(mask)
    indices = sort(indices)
    # print("indices", indices)

    p = gat(x, indices, block_size, block_stride, block_offset)
    print("gather results", p.shape)

    q = nn.functional.conv2d(p, w, stride=stride)

    conv_gt = gt_dict["conv_c"+str(channels)]
    # print(q, q.shape)

    assert np.array_equal(conv_gt, q.cpu().numpy()), \
        "Convolution results are incorrect"
    print("Conv results", q.shape)

    # x, y_base, indices, block_size, block_stride, block_offset
    y_base = torch.zeros(out_shape).cuda()

    y = scat(q, y_base, indices, block_size, block_stride)
    print("Scatter results", y.shape)

    scatter_gt = gt_dict["scatter_c"+str(channels)]

    print("Scatter results", y, y.shape)
    assert np.array_equal(scatter_gt, y.cpu().numpy())
    if np.array_equal(scatter_gt, y.cpu().numpy()):
        print("Sparse Scatter is correct!")


class SparseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_size, kernel_size, stride,
                 avg_pool=False, reduce_thresh=0, mask_size=(1, 1), add=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add = add

        self.reduce = ReduceMask(mask_size, reduce_thresh, block_size=block_size, kernel_size=kernel_size,
                                 stride=stride, padding="SAME", avg_pool=avg_pool)
        self.sparse_gather = SparseGather()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.sparse_scatter = SparseScatter(out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=(1, 1), add=add, atomic=add)

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, mask):
        # print("Before Gather", score.shape)
        indices, (block_size, block_stride, block_offset) = self.reduce(mask)
        dense_x = self.sparse_gather(x, indices,
                                     block_size, block_stride,
                                     block_offset)
        # print("dense_x: ", dense_x.shape)
        # print("After gather:", dense_x.shape)

        dense_x = self.conv1(dense_x)
        dense_x = self.bn1(dense_x)
        dense_x = self.relu(dense_x)
        dense_x = self.conv2(dense_x)
        dense_x = self.bn2(dense_x)
        dense_x = self.relu(dense_x)
        dense_x = self.conv3(dense_x)
        dense_x = self.bn3(dense_x)
        dense_y = self.relu(dense_x)
        # dense_y = dense_x

        # print("Before Scatter:", dense_y.shape)

        if self.add:
            # fix the dimensionality of the base tensor
            if self.in_channels != self.out_channels:
                ybase = self.downsample(x)
            else:
                ybase = x

        else:
            ybase = torch.zeros(x.shape[0],
                                dense_y.shape[1],
                                x.shape[2],
                                x.shape[3],
                                device=x.device)

        y = self.sparse_scatter(dense_y,
                                ybase,
                                indices, block_size, block_stride)
        # print("After scatter", score.shape)
        return y


def test_backward():
    channels = 2
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
    x.requires_grad = True
    mask = torch.from_numpy(mask[np.newaxis, :, :, :]).cuda()

    model = SparseBlock(in_channels=channels, out_channels=channels,
                        block_size=(3, 3), kernel_size=(3, 3), stride=(1, 1),
                        avg_pool=False, reduce_thresh=0, add=True, mask_size=mask.shape[2:4])
    model.cuda()

    print('x shape: ', x.shape)
    y = model(x, mask)
    print('y shape: ', y.shape)
    s = y.sum()
    s.backward()
    print("x grad:", x.grad, x.grad.shape)


if __name__ == "__main__":
    for c in [1, 2]:
        print("Testing with {} channels".format(c))
        test_sparse_gather(c)
        test_sparse_scatter(c)
    # test_backward()
