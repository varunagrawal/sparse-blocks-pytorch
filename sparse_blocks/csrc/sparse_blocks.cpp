#include "sparse_blocks.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("reducemask_forward", &reducemask_forward, "Reduce Mask Forward");
    // m.def("reduce_mask_backward", &reduce_mask_backward, "Reduce Mask Backward");
    m.def("sparse_gather_forward", &sparse_gather_forward, "Sparse Gather forward");
    m.def("sparse_gather_backward", &sparse_gather_backward, "Sparse Gather backward");
    // m.def("sparse_scatter_forward", &sparse_scatter_forward, "Sparse Scatter forward");
    // m.def("sparse_scatter_backward", &sparse_scatter_backward, "Sparse Scatter backward");
}