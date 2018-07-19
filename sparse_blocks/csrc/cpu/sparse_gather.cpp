#include <iostream>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <THC/THC.h>
#include <mutex>
#include <omp.h>

template <typename T>
void sparse_gather(const T *x, int N, int H, int W, int C, T *y,
                   int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
                   int numActive, const int *activeBlockIndices, bool transpose)
{
    const int R = bSzH, S = bSzW;
#pragma omp parallel for
    for (int ib = 0; ib < numActive; ib++)
    {
        int biN = activeBlockIndices[ib * 3 + 0];
        int biH = activeBlockIndices[ib * 3 + 1];
        int biW = activeBlockIndices[ib * 3 + 2];
        int h0 = bOffsH0 + biH * bStrH;
        int w0 = bOffsW0 + biW * bStrW;
        for (int intraBh = 0; intraBh < R; ++intraBh)
        {
            for (int intraBw = 0; intraBw < S; ++intraBw)
            {
                for (int cc = 0; cc < C; cc++)
                {
                    int hh = h0 + intraBh;
                    int ww = w0 + intraBw;
                    T readVal = 0.0f;
                    if (hh >= 0 && ww >= 0 && hh < H && ww < W)
                    {
                        readVal = x[biN * H * W * C + hh * W * C + ww * C + cc];
                    }

                    if (transpose) // output to gathered blocks in NCHW
                        y[ib * R * S * C + cc * R * S + intraBh * S + intraBw] = readVal;
                    else
                        y[ib * R * S * C + intraBh * S * C + intraBw * C + cc] = readVal;
                }
            }
        }
    }
}

at::Tensor sparse_gather_forward_cpu(const at::Tensor &x,
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

    bool transpose = true;

    int num_active = indices.size(0);

    at::Tensor y = at::zeros({num_active, C, blockH, blockW}, torch::CPU(at::kFloat));

    AT_DISPATCH_ALL_TYPES(x.type(), "sparse_gather_forward_cpu", ([&] {
                              sparse_gather<scalar_t>(
                                  x.data<scalar_t>(),
                                  N, H, W, C,
                                  y.data<scalar_t>(),
                                  bOffsH0, bOffsW0,
                                  blockH, blockW,
                                  blockStrH, blockStrW,
                                  num_active, indices.data<int>(),
                                  transpose);
                          }));

    return y;
}

at::Tensor sparse_gather_backward_cpu(const at::Tensor &grad_y)
{
    AT_ERROR("Backward pass of SparseGather is SparseScatter. Please directly call that.");
}
