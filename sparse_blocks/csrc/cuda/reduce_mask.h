#pragma once
#include <torch/torch.h>

at::Tensor reducemask_forward_cuda(const at::Tensor &mask,  // Mask Tensor
                                   int N,                   // Batch size
                                   int H,                   // Height of mask
                                   int W,                   // Width of mask
                                   float threshold,         // Threshold
                                   int bOffsH0,             // Block padding offset height, negative.
                                   int bOffsW0,             // Block padding offset width, negative.
                                   int blockH,              // Block size, height.
                                   int blockW,              // Block size, width.
                                   int blockCntH,
                                   int blockCntW,
                                   int blockStrH,           // Block stride, height.
                                   int blockStrW,           // Block stride, width.
                                   bool avg_pool);
