import torch
from torch import nn
from torchvision import models
from sparse_blocks import ReduceMask, SparseGather, SparseScatter
import pendulum

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logger = logging.Logger(__name__)


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
        logger.debug("ResidualBlock time:\t{}".format(
            (end-start).as_timedelta()))
        return x


class SparseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_size, kernel_size, stride,
                 avg_pool=False, mask_size=None, reduce_thresh=0, add=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add = add

        self.reduce = ReduceMask(mask_size, reduce_thresh, block_size=block_size, kernel_size=kernel_size,
                                 stride=stride, padding="SAME", avg_pool=avg_pool)
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

    def forward(self, x, mask):
        reduce_time = pendulum.now()
        indices, (block_size, block_stride, block_offset) = self.reduce(mask)
        logger.debug("ReduceMask time:\t{}".format(
            (pendulum.now() - reduce_time).as_timedelta()))

        gather_time = pendulum.now()
        dense_x = self.sparse_gather(x, indices,
                                     block_size, block_stride,
                                     block_offset)
        logger.debug("SparseGather time:\t{}".format(
            (pendulum.now() - gather_time).as_timedelta()))

        # logger.debug("dense_x:{}".format(dense_x.shape))

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
        #                     x.shape[3]).cuda()
        end = pendulum.now()
        logger.debug("SparseBlock Conv time:\t{}".format(
            (end-conv_start).as_timedelta()))

        scatter_time = pendulum.now()
        y = self.sparse_scatter(dense_y,
                                ybase,
                                indices, block_size, block_stride)
        logger.debug("SparseScatter time:\t{}".format(
            (pendulum.now() - scatter_time).as_timedelta()))

        return y


class Model(nn.Module):
    def __init__(self, output_channels, sparse=False):
        super().__init__()
        self.sparse = sparse
        self.output_channels = output_channels

        self.base = models.resnet101(pretrained=True)
        self.score4 = nn.ConvTranspose2d(in_channels=1024,
                                         out_channels=1024,
                                         kernel_size=4, stride=2, padding=1)
        if sparse:
            self.classifier = SparseBlock(in_channels=1024, out_channels=output_channels,
                                          block_size=(3, 3), kernel_size=(3, 3), stride=(1, 1),
                                          avg_pool=False, mask_size=(64, 64), reduce_thresh=0, add=True)
        else:
            self.classifier = ResidualBlock(1024, output_channels,
                                            kernel_size=3, padding=1)

    def forward(self, x, mask=None):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        res2 = x

        x = self.base.layer2(x)
        res3 = x

        x = self.base.layer3(x)
        res4 = x

        x = self.base.layer4(x)

        # upsample to 64x64
        score4 = self.score4(res4)

        # classif_start = pendulum.now()
        if self.sparse:
            y = self.classifier(score4, mask)
        else:
            y = self.classifier(score4)
        # logger.debug("Block time: ", (pendulum.now()-classif_start).microseconds)
        return y


data_size = 10
batch_size = 2


def get_mask():
    torch.manual_seed(123)
    mask = torch.rand(data_size, 1, 64, 64) > 0.9
    # mask = torch.zeros(data_size, 1, 64, 64)
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


def run(sparse=True):
    print("Sparsity=", sparse)

    model = Model(25, sparse=sparse).eval().cuda()

    total_time = 0

    mask = get_mask()
    x = torch.rand(batch_size, 3, 512, 512).cuda()

    for i in range(data_size):
        # x = torch.rand(batch_size, 3, 512, 512).cuda()

        start = pendulum.now()
        y = model(x, mask[i:i+batch_size].long().cuda())
        end = pendulum.now()

        # y.sum()
        y.add_(1)

        timedelta = end-start
        logger.warn("Model time\t\t{}".format(timedelta.as_timedelta()))
        if i > 0:
            total_time += timedelta.microseconds

    print("Total time (μs): ", total_time)
    avg_time = float(total_time)/data_size
    print("Average time (μs): ", avg_time)
    return avg_time


dense_time = run(sparse=False)
print("\n\n")
sparse_time = run(sparse=True)

print("\n\nSparse Speedup: ", dense_time/sparse_time)
