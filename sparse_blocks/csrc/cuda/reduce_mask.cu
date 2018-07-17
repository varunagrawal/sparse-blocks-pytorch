/*
   Sparse Blocks Network

   CUDA Kernel for Reduce Mask operation
*/

#include <iostream>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "cuda_helpers.h"


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


template <typename T>
__global__ void reducemask_forward_cuda_kernel(T* mask,
                                               int N,
                                               int H,
                                               int W,
                                               float threshold,
                                               unsigned int numBins,
                                               unsigned int binSize,
                                               int *bin_counts,
                                               int bOffsH0, int bOffsW0,
                                               int blockStrH, int blockStrW,   
                                               int blockCntH,
                                               int blockCntW,
                                               int blockH, int blockW,
                                               bool avg_pool,
                                               int *active_block_idxs)
{
    // int C = 1; // C is assumed to be 1
    const int roundedUpThreads = DIV_CEIL(blockH*blockW, warpLanes)*warpLanes;
    const int bH = blockH, bW = blockW;

    int blockW0 = bOffsW0 + blockStrW*blockIdx.x;
    int blockH0 = bOffsH0 + blockStrH*blockIdx.y;
    int n = blockIdx.z;

    // one thread per sparsity block pixel
    float mx = avg_pool ? 0.0f : -1e30;

    // allocate and initialize shmem for block reduce
    constexpr int maxBlockDim = 1024;
    assert(blockDim.x <= maxBlockDim);
    __shared__ float shmemx[maxBlockDim];
    for (int initOffs = 0; initOffs < maxBlockDim; initOffs += blockDim.x)
        if (initOffs + threadIdx.x < maxBlockDim)
            shmemx[initOffs+threadIdx.x] = avg_pool ? 0.0f : -1e30f;
    __syncthreads();

    // for large sparsity blocks we need multiple CUDA block loops
    for (int tOffs = 0; tOffs < roundedUpThreads; tOffs+=blockDim.x)
    {
        int tid = threadIdx.x + tOffs;
        const T* blockStartN = mask + n*H*W;
        float readVal = avg_pool ? 0.0f : -1e30f; // this value will be used to pad the warp
        if (tid < bH*bW) { // TODO: not needed?
            int woffs = tid % bW;
            int hoffs = tid / bW;
            unsigned bhh = hoffs + blockH0, bww = woffs + blockW0;
            if (bhh < H && bww < W)
                readVal = blockStartN[bhh*W + bww];
        }

        // actual number of threads is rounded up to 32 but padded with zeroes
        // warp reduce for all threads
        mx = avg_pool ? (mx + readVal) : max(mx, readVal);
        #pragma unroll
        for (int offset = warpLanes/2; offset > 0; offset /= 2) {
            float warped = __shfl_down(mx, offset);
            mx = avg_pool ? (mx + warped) : max(mx, warped);
        }

        // store (first elems from) warp reduces into shmem
        if (tid % warpLanes == 0) {
            int offs = tid/warpLanes; // tid includes tOffs
            int offsWrap = offs%blockDim.x;
            if (avg_pool)
                // atomics not needed here since we wrap around each blockDim.x
                shmemx[offsWrap] += mx;
            else
                shmemx[offsWrap] = max(shmemx[offsWrap], mx);
        }
        __syncthreads();
    } // tOffs

    // final reduce over all warps
    if (threadIdx.x == 0) {
        float mx1 = shmemx[0];
        // For sizes >= blockIdx.x we already reduced in the above loop
        const int numWarps = min(DIV_CEIL(bH*bW, warpLanes), blockIdx.x);
        #pragma unroll
        for (int iWarp = 1; iWarp < numWarps; iWarp++)
            mx1 = avg_pool ? (mx1 + shmemx[iWarp]) : max(mx1, shmemx[iWarp]);

        if (avg_pool)
            mx1 /= float(bH*bW);

        if (mx1 > threshold) {
            // now we have the maximums computed for each block
            // we need to write out the maximums, total over-threshold count across grid
            // at this point the number of blocks is grid size, so N*bCntH*bCntW
            // bad case scenario is say 4*64*64 (larger batch won't fit into memory)
            // so we can have ~16k blocks
            // need an efficient gmem reduction
            unsigned int blockIndex = n*bH*bW + blockIdx.y*bW + blockIdx.x;
            unsigned int myBin = ((blockIndex*100017+1234567)>>4) % numBins;
            unsigned int inBinOffs;
            // check for bin overflow
            while ((inBinOffs = atomicAdd(&bin_counts[myBin], unsigned(1))) >= binSize)
            {
                atomicSub(&bin_counts[myBin], unsigned(1));
                myBin++;
            }

            int offs = (myBin*binSize+inBinOffs)*3;
            active_block_idxs[offs+0] = blockIdx.z;
            active_block_idxs[offs+1] = blockIdx.y;
            active_block_idxs[offs+2] = blockIdx.x;
        } // if (mx1 > threshold)
    } // if (tid == 0)
}

at::Tensor reducemask_forward_cuda(const at::Tensor &mask,
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
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");
    
    int max_indices = N * blockCntH * blockCntW;
    
    unsigned int num_bins = 1;
    unsigned int binSize = (max_indices + num_bins - 1) / num_bins;

    // Number of indices of active blocks
    at::Tensor bin_counts = at::zeros({(int)num_bins}, torch::CUDA(at::kInt));
    
    // triples of [n, ih, iw] indices for active blocks.
    at::Tensor active_block_idxs = at::zeros({max_indices, 3}, torch::CUDA(at::kInt));
    
    // We can use ATen to initialize the block counters
    // zero_block_counters<<<1, 32>>>(num_bins, bin_counts.data<unsigned int>());
    
    dim3 grid(blockCntW, blockCntH, N);
    dim3 block(std::min(DIV_CEIL(blockH*blockW, 32)*32, 1024), 1, 1);
    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
    
    AT_DISPATCH_ALL_TYPES(mask.type(), "reducemask_forward", [&] {
        reducemask_forward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
            mask.data<scalar_t>(), 
            N, H, W,
            threshold,
            num_bins, binSize,
            bin_counts.data<int>(),
            bOffsH0, bOffsW0,
            blockStrH, blockStrW,
            blockCntH, blockCntW,
            blockH, blockW,
            avg_pool,
            active_block_idxs.data<int>());
    });

    THCudaCheck(cudaGetLastError());
    
    // We only want to return the valid indices, hence we index into them.
    at::Tensor idx = at::arange(0, at::Scalar(bin_counts[0]).to<int>(), torch::CUDA(at::kLong));
    return at::index(active_block_idxs, idx);
}

/****** 
We don't require gradients of the mask

template <typename T>
__global__ void reducemask_backward_cuda_kernel()
{

}
******/                                              
                                               