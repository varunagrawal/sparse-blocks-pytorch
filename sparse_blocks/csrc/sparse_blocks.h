#pragma once

#include <torch/torch.h>

// #include "cpu/reduce_mask.h"
#include "cpu/sparse_gather.h"
// #include "cpu/sparse_scatter.h"

#ifdef WITH_CUDA
#include "cuda/reduce_mask.h"
#include "cuda/sparse_gather.h"
#include "cuda/sparse_scatter.h"
#endif

at::Tensor reducemask_forward(const at::Tensor &mask,
                              int N,
                              int H,
                              int W,
                              float threshold,
                              int bOffsH0,
                              int bOffsW0,
                              int blockH,
                              int blockW,
                              int blockCntH,
                              int blockCntW,
                              int blockStrH,
                              int blockStrW,
                              bool avg_pool)
{
    if (mask.type().is_cuda())
    {
#ifdef WITH_CUDA
        return reducemask_forward_cuda(mask,
                                       N, H, W, threshold,
                                       bOffsH0, bOffsW0,
                                       blockH, blockW,
                                       blockCntH, blockCntW,
                                       blockStrH, blockStrW,
                                       avg_pool);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        // We should never need to come here
        // since for the CPU, we can just use the slower PyTorch version
        AT_ERROR("CPU version of ReduceMask should be handled by torch code, not C++ code");
    }
}

at::Tensor sparse_gather_forward(const at::Tensor &x, // should be NHWC
                                 const at::Tensor &indices,
                                 int blockH, int blockW,
                                 int blockStrH, int blockStrW,
                                 int bOffsH0, int bOffsW0)
{
    if (x.type().is_cuda())
    {
#ifdef WITH_CUDA
        // returns tensor of size (NCHW)
        // the kernel transposes the output
        return sparse_gather_forward_cuda(x, indices,
                                          blockH, blockW,
                                          blockStrH, blockStrW,
                                          bOffsH0, bOffsW0);
#else
        AT_ERROR("SparseGather not compiled with GPU support");
#endif
        // }
        // else
        // {
        //     return sparse_gather_forward_cpu(x);
    }
}

at::Tensor sparse_gather_backward(const at::Tensor &grad_y)
{
    if (grad_y.type().is_cuda())
    {
#ifdef WITH_CUDA
        return sparse_gather_backward_cuda(grad_y);
#else
        AT_ERROR("SparseGather not compiled with GPU support");
#endif
        // }
        // else
        // {
        //     return sparse_gather_backward_cpu(grad_y);
    }
}

at::Tensor sparse_scatter_forward(const at::Tensor &x, // should be NHWC
                                  const at::Tensor &indices,
                                  at::Tensor &ybase,
                                  int blockH, int blockW,
                                  int blockStrH, int blockStrW,
                                  int bOffsH0, int bOffsW0,
                                  bool add, bool atomic)
{
    if (x.type().is_cuda())
    {
#ifdef WITH_CUDA
        // returns tensor of size (NCHW)
        // the kernel transposes the output
        return sparse_scatter_forward_cuda(x, indices, ybase,
                                           blockH, blockW,
                                           blockStrH, blockStrW,
                                           bOffsH0, bOffsW0,
                                           add, atomic);
#else
        AT_ERROR("SparseScatter not compiled with GPU support");
#endif
        // }
        // else
        // {
        //     return sparse_scatter_forward_cpu(x);
    }
}

at::Tensor sparse_scatter_backward(const at::Tensor &grad_y)
{
    if (grad_y.type().is_cuda())
    {
#ifdef WITH_CUDA
        return sparse_scatter_backward_cuda(grad_y);
#else
        AT_ERROR("SparseScatter not compiled with GPU support");
#endif
        // }
        // else
        // {
        //     return sparse_scatter_backward_cpu(grad_y);
    }
}