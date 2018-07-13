#ifndef ZERO_BLOCK_COUNTERS_H
#define ZERO_BLOCK_COUNTERS_H

#include "helpers.h"

__device__ void zero_block_counters_kernel(unsigned int num_bins, unsigned int *bin_counts)
{
    // initialize bin_counts to 0
    int num_loops = DIV_CEIL(num_bins, gridDim.x);
    for (int i_block = 0; i_block < num_loops; ++i_block)
    {
        int writeIdx = (blockIdx.x * blockDim.x + threadIdx.x);
        writeIdx += (i_block * gridDim.x * blockDim.x);
        if (writeIdx < num_bins)
        {
            bin_counts[writeIdx] = 0;
        }
    }
}

__global__ void zero_block_counters(unsigned int num_bins, unsigned int *bin_counts)
{
    zero_block_counters_kernel(num_bins, bin_counts);
}

#endif