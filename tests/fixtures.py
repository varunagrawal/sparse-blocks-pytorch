import numpy as np
from numpy import array, float32


gt_dict = {
    "reduce_mask_c1": [],
    "gather_c1": array([[[[0.,  0.,  0.],
                          [0.,  1.,  2.],
                          [5.,  6.,  7.]]],
                        [[[0.,  0.,  0.],
                          [1.,  2.,  3.],
                            [6.,  7.,  8.]]],
                        [[[0.,  0.,  0.],
                          [2.,  3.,  4.],
                            [7.,  8.,  9.]]],
                        [[[0.,  0.,  1.],
                          [0.,  5.,  6.],
                            [0., 10., 11.]]],
                        [[[0.,  1.,  2.],
                          [5.,  6.,  7.],
                            [10., 11., 12.]]],
                        [[[1.,  2.,  3.],
                          [6.,  7.,  8.],
                            [11., 12., 13.]]],
                        [[[2.,  3.,  4.],
                          [7.,  8.,  9.],
                            [12., 13., 14.]]],
                        [[[0.,  5.,  6.],
                          [0., 10., 11.],
                            [0., 15., 16.]]],
                        [[[5.,  6.,  7.],
                          [10., 11., 12.],
                            [15., 16., 17.]]],
                        [[[6.,  7.,  8.],
                          [11., 12., 13.],
                            [16., 17., 18.]]],
                        [[[7.,  8.,  9.],
                          [12., 13., 14.],
                            [17., 18., 19.]]],
                        [[[0., 10., 11.],
                          [0., 15., 16.],
                            [0., 20., 21.]]],
                        [[[10., 11., 12.],
                          [15., 16., 17.],
                            [20., 21., 22.]]]], dtype=float32),
    "conv_c1": array([[[[21.]]],
                      [[[27.]]],
                      [[[33.]]],
                      [[[33.]]],
                      [[[54.]]],
                      [[[63.]]],
                      [[[72.]]],
                      [[[63.]]],
                      [[[99.]]],
                      [[[108.]]],
                      [[[117.]]],
                      [[[93.]]],
                      [[[144.]]]], dtype=float32),
    "scatter_c1": np.array([[[[0.,  21.,  27.,  33.,   0.],
                              [33.,  54.,  63.,  72.,   0.],
                              [63.,  99., 108., 117.,   0.],
                              [93., 144.,   0.,   0.,   0.],
                              [0.,   0.,   0.,   0.,   0.]]]], dtype=np.float32),
    "reduce_mask_c2": [],
    "gather_c2": array([[[[0.,  0.,  0.],
                          [0.,  2.,  4.],
                          [10., 12., 14.]],
                         [[0.,  0.,  0.],
                          [1.,  3.,  5.],
                          [11., 13., 15.]]],
                        [[[0.,  0.,  0.],
                          [2.,  4.,  6.],
                            [12., 14., 16.]],
                         [[0.,  0.,  0.],
                            [3.,  5.,  7.],
                            [13., 15., 17.]]],
                        [[[0.,  0.,  0.],
                          [4.,  6.,  8.],
                            [14., 16., 18.]],
                         [[0.,  0.,  0.],
                            [5.,  7.,  9.],
                            [15., 17., 19.]]],
                        [[[0.,  0.,  2.],
                          [0., 10., 12.],
                            [0., 20., 22.]],
                         [[0.,  1.,  3.],
                            [0., 11., 13.],
                            [0., 21., 23.]]],
                        [[[0.,  2.,  4.],
                          [10., 12., 14.],
                            [20., 22., 24.]],
                         [[1.,  3.,  5.],
                            [11., 13., 15.],
                            [21., 23., 25.]]],
                        [[[2.,  4.,  6.],
                          [12., 14., 16.],
                            [22., 24., 26.]],
                         [[3.,  5.,  7.],
                            [13., 15., 17.],
                            [23., 25., 27.]]],
                        [[[4.,  6.,  8.],
                          [14., 16., 18.],
                            [24., 26., 28.]],
                         [[5.,  7.,  9.],
                            [15., 17., 19.],
                            [25., 27., 29.]]],
                        [[[0., 10., 12.],
                          [0., 20., 22.],
                            [0., 30., 32.]],
                         [[0., 11., 13.],
                            [0., 21., 23.],
                            [0., 31., 33.]]],
                        [[[10., 12., 14.],
                          [20., 22., 24.],
                            [30., 32., 34.]],
                         [[11., 13., 15.],
                            [21., 23., 25.],
                            [31., 33., 35.]]],
                        [[[12., 14., 16.],
                          [22., 24., 26.],
                            [32., 34., 36.]],
                         [[13., 15., 17.],
                            [23., 25., 27.],
                            [33., 35., 37.]]],
                        [[[14., 16., 18.],
                          [24., 26., 28.],
                            [34., 36., 38.]],
                         [[15., 17., 19.],
                            [25., 27., 29.],
                            [35., 37., 39.]]],
                        [[[0., 20., 22.],
                          [0., 30., 32.],
                            [0., 40., 42.]],
                         [[0., 21., 23.],
                            [0., 31., 33.],
                            [0., 41., 43.]]],
                        [[[20., 22., 24.],
                          [30., 32., 34.],
                            [40., 42., 44.]],
                         [[21., 23., 25.],
                            [31., 33., 35.],
                            [41., 43., 45.]]]], dtype=float32),
    "conv_c2": np.array([[[[90.]],
                          [[90.]]],
                         [[[114.]],
                          [[114.]]],
                         [[[138.]],
                          [[138.]]],
                         [[[138.]],
                          [[138.]]],
                         [[[225.]],
                          [[225.]]],
                         [[[261.]],
                          [[261.]]],
                         [[[297.]],
                          [[297.]]],
                         [[[258.]],
                          [[258.]]],
                         [[[405.]],
                          [[405.]]],
                         [[[441.]],
                          [[441.]]],
                         [[[477.]],
                          [[477.]]],
                         [[[378.]],
                          [[378.]]],
                         [[[585.]],
                          [[585.]]]], dtype=np.float32),
    "scatter_c2": array([[[[0.,  90., 114., 138.,   0.],
                           [138., 225., 261., 297.,   0.],
                           [258., 405., 441., 477.,   0.],
                           [378., 585.,   0.,   0.,   0.],
                           [0.,   0.,   0.,   0.,   0.]],
                          [[0.,  90., 114., 138.,   0.],
                           [138., 225., 261., 297.,   0.],
                           [258., 405., 441., 477.,   0.],
                           [378., 585.,   0.,   0.,   0.],
                           [0.,   0.,   0.,   0.,   0.]]]], dtype=float32)

}