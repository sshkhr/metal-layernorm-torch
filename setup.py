import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

def get_extensions():
    extensions = []
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # CRITICAL: Patch UnixCCompiler to recognize .mm files
        from distutils.unixccompiler import UnixCCompiler
        if '.mm' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.mm')
            UnixCCompiler.language_map['.mm'] = 'objc'

        ext = CppExtension(
            name='layernorm_metal._C',
            sources=['src/dispatch.mm'],
            extra_compile_args={'cxx': [
                '-std=c++17',
                '-ObjC++',
                '-Wall',
            ]},
            extra_link_args=[
                '-framework', 'Metal',
                '-framework', 'Foundation',
            ],
        )
        extensions.append(ext)
    return extensions

setup(
    name='layernorm_metal',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={'layernorm_metal': ['kernels/*.metal']},
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
)