#pragma once
#include <torch/torch.h>

at::Tensor sparse_gather_forward_cpu(const at::Tensor &x,       // should be NCHW
                                     const at::Tensor &indices, //
                                     int blockH, int blockW,
                                     int blockStrH, int blockStrW,
                                     int bOffsH0, int bOffsW0,
                                     bool tranpose);

at::Tensor sparse_gather_backward_cpu(const at::Tensor &grad_y);
