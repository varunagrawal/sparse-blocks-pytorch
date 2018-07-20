# Sparse Blocks Pytorch

PyTorch CUDA Extension for Sparse Blocks as specified in the [Sparse Blocks Network paper](https://arxiv.org/abs/1801.02108).


**NOTE**
    
    While the original paper uses the "NHWC" data layout for leveraging memory locality, the "NHWC" layout is specific for TensorFlow.
    Pytorch's use of tensor Storage means that the memory locality will be in the "WC" dimensions, hence this extension expects all input and output in the "NCHW" layout. 
