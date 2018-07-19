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


at::Tensor sparse_scatter_forward_cuda(const at::Tensor &x, 
                                       const at::Tensor &indices,
                                       at::Tensor &ybase,
                                       int blockH, int blockW,
                                       int blockStrH, int blockStrW,
                                       int bOffsH0, int bOffsW0,
                                       bool add, bool atomic)
{
    // We need the dimensions of the original feature map to scatter to.
    int N = ybase.size(0);
    int C = ybase.size(1);
    int H = ybase.size(2);
    int W = ybase.size(3);

    // flag to indicate that x is in NCHW format
    bool transpose = true;

    int num_active = indices.size(0);

    if (!atomic && add) 
    {
        if (blockStrH >= blockH && blockStrW >= blockW)
        {
            AT_ERROR("Only non-overlapping blocks are supported with add=True, atomic=False");
        }
    }

    at::Tensor y = at::zeros(ybase.sizes(), torch::CUDA(at::kFloat));
    y.copy_(ybase);
    
    bool hasInst = false;

    cudaStream_t stream = at::globalContext().getCurrentCUDAStream();

    LaunchParams lp(C, blockH, blockW, num_active);
 
    #define CALL(RR, CC1, addt, transt) \
        if (blockH == RR && blockW == RR && lp.fittingC1 == CC1 && atomic == false) { \
            hasInst = true; \
            AT_DISPATCH_ALL_TYPES(x.type(), "sparse scatter", [&] { \
                blockScatterTiled0<scalar_t, 512, RR, COMPUTE_R1(RR), RR, CC1, addt, transt, false> \
                    <<<lp.grid, lp.block, lp.shmemSize, stream>>>( \
                        x.data<scalar_t>(), indices.data<int>(), \
                        y.data<scalar_t>(), \
                        N, H, W, C, \
                        bOffsH0, bOffsW0, blockStrH, blockStrW); \
            }); \
        } else

    #define SIZE_TEMPLATES(addt, transpt, CCC) \
        CALL( 1, CCC, addt, transpt) \
        CALL( 2, CCC, addt, transpt) \
        CALL( 3, CCC, addt, transpt) \
        CALL( 4, CCC, addt, transpt) \
        CALL( 5, CCC, addt, transpt) \
        CALL( 6, CCC, addt, transpt) \
        CALL( 7, CCC, addt, transpt) \
        CALL( 8, CCC, addt, transpt) \
        CALL( 9, CCC, addt, transpt) \
        CALL(10, CCC, addt, transpt) \
        CALL(11, CCC, addt, transpt) \
        CALL(12, CCC, addt, transpt) \
        CALL(13, CCC, addt, transpt) \
        CALL(14, CCC, addt, transpt) \
        CALL(15, CCC, addt, transpt) \
        CALL(16, CCC, addt, transpt) \
        CALL(17, CCC, addt, transpt) \
        CALL(18, CCC, addt, transpt) \
        CALL(19, CCC, addt, transpt) \
        CALL(20, CCC, addt, transpt) \
        CALL(21, CCC, addt, transpt) \
        CALL(22, CCC, addt, transpt) \
        CALL(23, CCC, addt, transpt) \
        CALL(24, CCC, addt, transpt) \
        CALL(25, CCC, addt, transpt) \
        CALL(26, CCC, addt, transpt) \
        CALL(27, CCC, addt, transpt) \
        CALL(28, CCC, addt, transpt) \
        CALL(29, CCC, addt, transpt) \
        CALL(30, CCC, addt, transpt) \
        CALL(31, CCC, addt, transpt) \
        CALL(32, CCC, addt, transpt) \
        CALL(33, CCC, addt, transpt) \
        CALL(34, CCC, addt, transpt) \
        CALL(41, CCC, addt, transpt) \
        CALL(48, CCC, addt, transpt) \
        CALL(63, CCC, addt, transpt) \
        CALL(64, CCC, addt, transpt) \
        CALL(65, CCC, addt, transpt) \
        CALL(81, CCC, addt, transpt) \
            hasInst = false;

    // We assume transpose is always true 
    // since we want the output to be of size (NCHW)
    if (!add) {
        if (lp.fittingC1 >= 32) {
            SIZE_TEMPLATES(false, true, 32)
        } else if (lp.fittingC1 == 16) {
            SIZE_TEMPLATES(false, true, 16)
        } else if (lp.fittingC1 == 24) {
            SIZE_TEMPLATES(false, true, 24)
        }
    } else if (add) {
        if (lp.fittingC1 >= 32) {
            SIZE_TEMPLATES(true, true, 32)
        } else if (lp.fittingC1 == 16) {
            SIZE_TEMPLATES(true, true, 16)
        } else if (lp.fittingC1 == 24) {
            SIZE_TEMPLATES(true, true, 24)
        }
    }

    if (!hasInst) {
        //printf("scatter, C, bSzH, bSzW=%d, %d, %d, fittingC1=%d\n", C, bSzH, bSzW, lp.fittingC1);
        AT_DISPATCH_ALL_TYPES(x.type(), "sparse scatter", [&] {
            blockScatterTiled1<scalar_t, 512><<<lp.grid, lp.block, lp.shmemSize, stream>>>(
                x.data<scalar_t>(), indices.data<int>(),
                y.data<scalar_t>(), 
                N, H, W, C, 
                bOffsH0, bOffsW0, blockStrH, blockStrW,
                blockH, lp.bSzH1, blockW, lp.fittingC1, add, transpose, atomic);
        });
    }
    #undef SIZE_TEMPLATES
    #undef CALL
    THCudaCheck(cudaGetLastError());

    return y;
}


at::Tensor sparse_scatter_backward_cuda(const at::Tensor &grad_y)
{
    AT_ERROR("Backward pass of SparseScatter is SparseGather. Please directly call that.");
}