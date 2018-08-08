import torch
from torch import nn
from torchvision import models
from sparse_blocks import ReduceMask, SparseGather, SparseScatter
import pendulum

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("sbnet")

# torch.backends.cudnn.enabled = False

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        start = pendulum.now()
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.in_channels != self.out_channels:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.relu(x)

        end = pendulum.now()
        logger.debug("ResidualBlock time:\t{}".format((end-start).as_timedelta()))
        return x


class SparseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_size, kernel_size, stride,
                 avg_pool=False, mask_size=None, reduce_thresh=0, add=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add = add

        self.sparse_gather = SparseGather()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.sparse_scatter = SparseScatter(out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=(1, 1), add=add, atomic=add, padding='SAME')

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, mask, indices, block_size, block_stride, block_offset):
        # reduce_time = pendulum.now()
        # indices, (block_size, block_stride, block_offset) = self.reduce(mask)
        # logger.debug("ReduceMask time:\t{}".format((pendulum.now() - reduce_time).as_timedelta()))

        gather_time = pendulum.now()
        dense_x = self.sparse_gather(x, indices,
                                     block_size, block_stride,
                                     block_offset)
        logger.debug("SparseGather time:\t{}".format((pendulum.now() - gather_time).as_timedelta()))

        logger.debug("dense_x:{}".format(dense_x.shape))

        conv_start = pendulum.now()

        dense_x = self.conv1(dense_x)
        dense_x = self.bn1(dense_x)
        dense_y = self.relu(dense_x)
        dense_x = self.conv2(dense_x)
        dense_x = self.bn2(dense_x)
        dense_y = self.relu(dense_x)
        dense_x = self.conv3(dense_x)
        dense_x = self.bn3(dense_x)

        if self.in_channels != self.out_channels:
            ybase = self.downsample(x)
        else:
            ybase = x

        # ybase = torch.zeros(x.shape[0],
        #                     dense_y.shape[1],
        #                     x.shape[2],
        #                     x.shapemask[i:i+batch_size].long().cuda()[3]).cuda()
        end = pendulum.now()
        logger.debug("SparseBlock Conv time:\t{}".format((end-conv_start).as_timedelta()))

        scatter_time = pendulum.now()
        y = self.sparse_scatter(dense_y,
                                ybase,
                                indices, block_size, block_stride)
        logger.debug("SparseScatter time:\t{}".format((pendulum.now() - scatter_time).as_timedelta()))

        return y


class ResidualUnits(nn.Module):
    """
    Residual Units as described in the SBNet paper
    """
    def __init__(self, n, in_channels, out_channels):
        super().__init__()
        self.res = nn.ModuleList()
        for i in range(n):
            self.res.append(ResidualBlock(in_channels[i], out_channels[i]))
        self.n = n

    def forward(self, x):
        # start = pendulum.now()
        for i in range(self.n):
            x = self.res[i](x)

        # end = pendulum.now()
        # logger.info("Residual Unit time for n={}:".format(self.n), (end-start).as_timedelta())
        return x


class SparseResidualUnits(nn.Module):
    """
    Sparse Residual Units as described in the SBNet paper
    """
    def __init__(self, n, in_channels, out_channels):
        super().__init__()
        self.reduce = ReduceMask((512, 512), 0, block_size=(3, 3), kernel_size=(3,3),
                                 stride=(1,1), padding="SAME", avg_pool=False)
        
        self.res = nn.ModuleList()
        for i in range(n):
            self.res.append(SparseBlock(in_channels[i], out_channels[i], 
                                        block_size=(3,3), kernel_size=(3,3), stride=(1,1),
                                        avg_pool=False, mask_size=(512, 512), reduce_thresh=0, add=True))
        self.n = n

    def forward(self, x, mask):
        indices, (block_size, block_stride, block_offset) = self.reduce(mask)
        # start = pendulum.now()
        for i in range(self.n):
            x = self.res[i](x, mask, indices, block_size, block_stride, block_offset)

        # end = pendulum.now()
        # logger.info("Residual Unit time for n={}:".format(self.n), (end-start).as_timedelta())
        return x


data_size = 2
batch_size = 1


def get_mask(size=(64, 64)):
    torch.manual_seed(123)
    mask = torch.rand(data_size, 1, size[0], size[1]) > 0.9
    # mask = torch.zeros(data_size, 1, size, size)
    # mask[:, :, 0:10, 3:23] = 1
    # mask[:, :, 33:55, 47:57] = 1
    
    # mask = torch.zeros(data_size, 1, 16, 16)
    # mask[:, :, 0:16:8, 0:16:8] = 1
    # mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode='nearest')

    # mask = torch.zeros(data_size, 1, 64, 64)
    # mask[:, :, 0:10, 0:40] = 1

    print("Sparsity %= ", mask.sum().float()*100 / mask.numel())
    return mask


print("Batch Size = ", batch_size)


def run_dense(size):
    print("Sparse=False")

    # model = Model(25, sparse=sparse).eval().cuda()
    model = ResidualUnits(6, (192, 48, 48, 48, 48, 48), (48, 48, 48, 48, 48, 192)).eval().cuda()
    total_time = 0

    mask = get_mask()

    # x = torch.rand(batch_size, 1024, 64, 64).cuda()
    x = torch.rand(batch_size, 192, size[0], size[1]).cuda()

    for i in range(data_size):
        # x = torch.rand(batch_size, 3, 512, 512).cuda()

        m = mask[i:i+batch_size].long().cuda()

        start = pendulum.now()
        y = model(x)
        end = pendulum.now()

        y.sum()
        y.add_(1)

        timedelta = end-start
        logger.info("Model time\t\t{}".format(timedelta.as_timedelta()))
        if i > 0:
            total_time += timedelta.microseconds

    print("Total time (μs): ", total_time)
    avg_time = float(total_time)/data_size
    print("Average time (μs): ", avg_time)
    return avg_time


def run_sparse(size):
    print("Sparse=True")

    model = SparseResidualUnits(6, (192, 48, 48, 48, 48, 48), (48, 48, 48, 48, 48, 192)).eval().cuda()
    total_time = 0

    mask = get_mask(size=size)
    # x = torch.rand(batch_size, 1024, 64, 64).cuda()
    x = torch.rand(batch_size, 192, size[0], size[1]).cuda()

    for i in range(data_size):
        # x = torch.rand(batch_size, 3, 512, 512).cuda()

        m = mask[i:i+batch_size].long().cuda()

        start = pendulum.now()
        y = model(x, m)
        end = pendulum.now()

        y.sum()
        y.add_(1)

        timedelta = end-start
        logger.info("Model time\t\t{}".format(timedelta.as_timedelta()))
        if i >= 0:
            total_time += timedelta.microseconds

    print("Total time (μs): ", total_time)
    avg_time = float(total_time)/data_size
    print("Average time (μs): ", avg_time)
    return avg_time

size = (50, 50)
dense_time = run_dense(size)
print("\n\n")
sparse_time = run_sparse(size)

print("\n\nSparse Speedup: ", dense_time/sparse_time)
