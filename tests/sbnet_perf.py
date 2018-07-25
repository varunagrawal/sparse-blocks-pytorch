import torch
from torch import nn
from torchvision import models
from sparse_blocks import ReduceMask, SparseGather, SparseScatter
import pendulum


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
        # reduce_time = pendulum.now()
        indices, (block_size, block_stride, block_offset) = self.reduce(mask)
        # print("ReduceMask time: ", (pendulum.now() - reduce_time).microseconds)

        # gather_time = pendulum.now()
        dense_x = self.sparse_gather(x, indices,
                                     block_size, block_stride,
                                     block_offset)
        # print("SparseGather time: ", (pendulum.now() - gather_time).microseconds)

        # print("dense_x:", dense_x.shape)

        # conv_start = pendulum.now()
        dense_x = self.conv1(dense_x)
        dense_x = self.bn1(dense_x)
        dense_y = self.relu(dense_x)
        dense_x = self.conv2(dense_x)
        dense_x = self.bn2(dense_x)
        dense_y = self.relu(dense_x)
        dense_x = self.conv3(dense_x)
        dense_x = self.bn3(dense_x)
        # print("Conv time: ", (pendulum.now()-conv_start).microseconds)

        if self.in_channels != self.out_channels:
            ybase = self.downsample(x)
        else:
            ybase = x

        # start = pendulum.now()
        # ybase = torch.zeros(x.shape[0],
        #                     dense_y.shape[1],
        #                     x.shape[2],
        #                     x.shape[3]).cuda()
        # print("instantiation time: ", (pendulum.now()-start).microseconds)

        # scatter_time = pendulum.now()
        y = self.sparse_scatter(dense_y,
                                ybase,
                                indices, block_size, block_stride)
        # print("SparseScatter time: ", (pendulum.now() - scatter_time).microseconds)

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
        # print("Block time: ", (pendulum.now()-classif_start).microseconds)
        return y


torch.manual_seed(123)

batch_size = 100
data = torch.rand(batch_size, 3, 512, 512)
mask = torch.rand(batch_size, 1, 64, 64) > 0.9
# mask = torch.zeros(batch_size, 1, 64, 64)
# mask[:, :, 0:10, 0:40] = 1
print("Sparsity %= ", mask.sum().float()*100 / mask.numel())


def run(sparse=True):
    print("Sparsity=", sparse)

    model = Model(125, sparse=sparse).cuda()

    start = pendulum.now()

    for i in range(batch_size):
        x = data[i].unsqueeze(0).cuda()
        y = model(x, mask[i].unsqueeze(0).long().cuda())

    end = pendulum.now()

    timedelta = end-start
    print("Average time (ms): ", timedelta.microseconds/batch_size)
    return timedelta.microseconds/batch_size


dense_time = run(sparse=False)
print("\n\n")
sparse_time = run(sparse=True)

print("\n\nSparse Speedup: ", dense_time/sparse_time)
