#pragma once
#include <torch/torch.h>

at::Tensor sparse_scatter_forward_cpu(const at::Tensor &x,
                                      const at::Tensor &indices,
                                      at::Tensor &ybase,
                                      int blockH, int blockW,
                                      int blockStrH, int blockStrW,
                                      int bOffsH0, int bOffsW0,
                                      bool add, bool atomic);