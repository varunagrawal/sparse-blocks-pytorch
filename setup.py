import os
import glob

from setuptools import setup, find_packages

import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extension_dir = os.path.join(this_dir, 'sparse_blocks', 'csrc')

    main_file = glob.glob(os.path.join(extension_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extension_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extension_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]

    sources = [os.path.join(extension_dir, s) for s in sources]

    include_dirs = [extension_dir]

    ext_modules = [
        extension(
            name='sparse_blocks._C',
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            # need to set NVIDIA compute capability flags to compile correctly
            # also set `-w` to disable warnings in the output during compilation.
            extra_compile_args={'cxx': ['-g', '-w'],
                                'nvcc': ['-gencode=arch=compute_61,code=sm_61', '-w']}
        )
    ]

    return ext_modules


requirements = [
    'torch',
]

setup(
    # Metadata
    name='sparse-blocks',
    version="1",
    author='Varun Agrawal',
    author_email='varunagrawal@gatech.edu',
    url='https://varunagrawal.github.io',
    description='Building Blocks for Sparse Blocks Network',
    long_description="",
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,

    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension}
)
