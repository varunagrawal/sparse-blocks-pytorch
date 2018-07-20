#pragma once
#include <torch/torch.h>

at::Tensor sparse_gather_forward_cuda(const at::Tensor &x,
                                      const at::Tensor &indices,
                                      int blockH, int blockW,
                                      int blockStrH, int blockStrW,
                                      int bOffsH0, int bOffsW0,
                                      bool tranpose);

at::Tensor sparse_gather_backward_cuda(const at::Tensor &grad_y);