/*
   Sparse Blocks Network

   CUDA Kernel for Sparse Gather operation
*/

#include <iostream>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "cuda_helpers.h"
#include "block_tile.cu.h"


#define COMPUTE_R1(RR) ((RR) < 7 ? ((RR) == 1 ? 1 : 2) : 4)

struct LaunchParams {
    dim3 block, grid;
    int shmemSize;
    int bSzH1;
    int fittingC1;
    enum { MAX_SHMEM = 24*1024 };
    LaunchParams(int C, int bSzH, int bSzW, int numActive)
    {
        fittingC1 = std::min(32, C);
        bSzH1 = COMPUTE_R1(bSzH);
        while ((shmemSize = (fittingC1+1)*bSzH1*bSzW*sizeof(float)) > MAX_SHMEM)
            fittingC1--;
        assert(fittingC1 >= 1);
        assert(bSzH1*bSzW*(fittingC1+1)*sizeof(float) <= MAX_SHMEM);
        block = dim3(512, 1, 1);
        grid = dim3(numActive, DIV_CEIL(C, fittingC1), DIV_CEIL(bSzH, bSzH1));
    }
};

at::Tensor sparse_gather_forward_cuda(const at::Tensor &x,
                                      const at::Tensor &indices,
                                      int blockH, int blockW,
                                      int blockStrH, int blockStrW,
                                      int bOffsH0, int bOffsW0)
{
    // Assume input is NHWC to leverage memory locality.
    int N = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    int C = x.size(3);
 
    int bin_count = indices.size(0);

    LaunchParams lp(C, blockH, blockW, bin_count);
    bool hasInst = false;

    at::Tensor y = at::zeros({bin_count, C, blockH, blockW}, torch::CUDA(at::kFloat));
    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();
        
    #define CALL(RR, CC1, trans) \
        if (blockH == RR && blockW == RR && lp.fittingC1 == CC1) { \
            hasInst = true; \
            AT_DISPATCH_ALL_TYPES(x.type(), "sparse gather", [&] { \
                blockGatherTiled0<scalar_t, 512, RR, COMPUTE_R1(RR), RR, CC1, trans><<<lp.grid, lp.block, lp.shmemSize, stream>>>( \
                    x.data<scalar_t>(), indices.data<const short>(), \
                    y.data<scalar_t>(), \
                    N, H, W, C, \
                    bOffsH0, bOffsW0, blockStrH, blockStrW); \
            }); \
        } else

    #define SIZE_TEMPLATES(transt, CCC) \
        CALL( 1, CCC, transt) \
        CALL( 2, CCC, transt) \
        CALL( 3, CCC, transt) \
        CALL( 4, CCC, transt) \
        CALL( 5, CCC, transt) \
        CALL( 6, CCC, transt) \
        CALL( 7, CCC, transt) \
        CALL( 8, CCC, transt) \
        CALL( 9, CCC, transt) \
        CALL(10, CCC, transt) \
        CALL(11, CCC, transt) \
        CALL(12, CCC, transt) \
        CALL(13, CCC, transt) \
        CALL(14, CCC, transt) \
        CALL(15, CCC, transt) \
        CALL(16, CCC, transt) \
        CALL(17, CCC, transt) \
        CALL(18, CCC, transt) \
        CALL(19, CCC, transt) \
        CALL(20, CCC, transt) \
        CALL(21, CCC, transt) \
        CALL(22, CCC, transt) \
        CALL(23, CCC, transt) \
        CALL(24, CCC, transt) \
        CALL(25, CCC, transt) \
        CALL(26, CCC, transt) \
        CALL(27, CCC, transt) \
        CALL(28, CCC, transt) \
        CALL(29, CCC, transt) \
        CALL(30, CCC, transt) \
        CALL(31, CCC, transt) \
        CALL(32, CCC, transt) \
        CALL(33, CCC, transt) \
        CALL(34, CCC, transt) \
        CALL(41, CCC, transt) \
        CALL(48, CCC, transt) \
        CALL(63, CCC, transt) \
        CALL(64, CCC, transt) \
        CALL(65, CCC, transt) \
        CALL(81, CCC, transt) \
            { hasInst = false; }

    if (lp.fittingC1 >= 32) {
        SIZE_TEMPLATES(true, 32)
    } else if (lp.fittingC1 == 16) {
        SIZE_TEMPLATES(true, 16)
    } else if (lp.fittingC1 == 24) {
        SIZE_TEMPLATES(true, 24)
    }
    
    if (!hasInst)
    {
        //printf("gather, C, bSzH, bSzW=%d, %d, %d, fittingC1=%d\n", C, bSzH, bSzW, lp.fittingC1);
        AT_DISPATCH_ALL_TYPES(x.type(), "sparse gather", [&] {
            blockGatherTiled1<scalar_t, 512><<<lp.grid, lp.block, lp.shmemSize, stream>>>(
                x.data<scalar_t>(), indices.data<const short>(),
                y.data<scalar_t>(), 
                N, H, W, C, 
                bOffsH0, bOffsW0, blockStrH, blockStrW,
                blockH, lp.bSzH1, blockW, lp.fittingC1, true);    
        });
    }
    #undef SIZE_TEMPLATES
    #undef CALL
    THCudaCheck(cudaGetLastError());

    return y;
}

at::Tensor sparse_gather_backward_cuda(const at::Tensor &grad_y)
{
    AT_ERROR("Backward pass of SparseGather is SparseScatter. Please directly call that.");
}