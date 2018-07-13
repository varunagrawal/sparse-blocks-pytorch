/*
   Sparse Blocks Network
*/

#include <ATen/ATen.h>
#include "reduce_mask.h"
#include "zero_block_counters.cu.h"
#include "reduce_mask.cu.h"
#include "helpers.h"


template <typename T>
__global__ void reducemask_forward_cuda_kernel(

)
{

}

at::Tensor reducemask_forward_cuda(const at::Tensor& mask)
{
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");
}

template <typename T>
__global__ void reducemask_backward_cuda_kernel(

)
{

}
