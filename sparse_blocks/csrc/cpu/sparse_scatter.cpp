#include <iostream>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <THC/THC.h>
#include <mutex>
#include <omp.h>

template <typename T>
void sparse_scatter(const T *x, int N, int H, int W, int C, T *y,
                    int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
                    int numActive, const int *activeBlockIndices, bool add, bool transpose, bool atomic)
{
    omp_lock_t writeLock;
    omp_init_lock(&writeLock);

    const int R = bSzH, S = bSzW;
#pragma omp parallel for
    for (int ib = 0; ib < numActive; ib++)
    {
        int biN = activeBlockIndices[ib * 3 + 0];
        int biH = activeBlockIndices[ib * 3 + 1];
        int biW = activeBlockIndices[ib * 3 + 2];

        for (int intraBh = 0; intraBh < R; ++intraBh)
        {
            for (int intraBw = 0; intraBw < S; ++intraBw)
            {
                for (int cc = 0; cc < C; cc++)
                {
                    int h0 = bOffsH0 + biH * bStrH;
                    int w0 = bOffsW0 + biW * bStrW;
                    int hh = h0 + intraBh;
                    int ww = w0 + intraBw;
                    T readVal;
                    if (transpose)
                        readVal = x[ib * R * S * C + cc * R * S + intraBh * S + intraBw];
                    else
                        readVal = x[ib * R * S * C + intraBh * S * C + intraBw * C + cc];

                    if (hh >= 0 && ww >= 0 && hh < H && ww < W)
                    {
                        if (add)
                        {
                            omp_set_lock(&writeLock);
                            y[biN * H * W * C + hh * W * C + ww * C + cc] += readVal;
                            omp_unset_lock(&writeLock);
                        }
                        else
                        {
                            // Write in NCHW format
                            y[biN * H * W * C + cc * H * W + hh * W + ww] = readVal;
                            // y[biN * H * W * C + hh * W * C + ww * C + cc] = readVal;
                        }
                    }
                }
            }
        }
    }
}

at::Tensor sparse_scatter_forward_cpu(const at::Tensor &x,
                                      const at::Tensor &indices,
                                      at::Tensor &ybase, // ybase is NCHW
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

    bool transpose = true;

    int num_active = indices.size(0);

    if (!atomic && add)
    {
        if (blockStrH >= blockH && blockStrW >= blockW)
        {
            AT_ERROR("Only non-overlapping blocks are supported with add=True, atomic=False");
        }
    }

    int n = ybase.size(0);
    at::Tensor y = at::zeros(ybase.sizes(), torch::CPU(at::kFloat));
    y.copy_(ybase);

    AT_DISPATCH_ALL_TYPES(x.type(), "sparse_scatter_forward_cpu", ([&] {
                              sparse_scatter<scalar_t>(
                                  x.data<scalar_t>(),
                                  N, H, W, C,
                                  y.data<scalar_t>(),
                                  bOffsH0, bOffsW0,
                                  blockH, blockW,
                                  blockStrH, blockStrW,
                                  num_active, indices.data<int>(),
                                  add, transpose, atomic);
                          }));

    return y;
}

at::Tensor sparse_scatter_backward_cpu(at::Tensor grad_y)
{
    AT_ERROR("Backward pass of SparseScatter is SparseGather. Please directly call that.");
}