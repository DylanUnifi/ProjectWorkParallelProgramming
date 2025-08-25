# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import sysconfig
import os

class get_include:
    """Lazy import helpers for pybind11 and numpy includes."""
    @staticmethod
    def pybind11():
        import pybind11
        return pybind11.get_include()

    @staticmethod
    def numpy():
        import numpy as np
        return np.get_include()

def has_flag(compiler, flagname):
    """Return True if a flag is supported by the compiler."""
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main(){return 0;}')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except Exception:
            return False
    return True

def cpp_std_flag(compiler):
    for flag in ['-std=c++17', '-std=c++14', '/std:c++17', '/std:c++14']:
        if has_flag(compiler, flag):
            return flag
    return ''

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc', '/openmp'],
        'unix': ['-O3', '-fopenmp', '-ffast-math', '-fPIC'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        # C++ standard
        std_flag = cpp_std_flag(self.compiler)
        if std_flag:
            opts.append(std_flag)

        # march=native if supported
        if ct == 'unix' and has_flag(self.compiler, '-march=native'):
            opts.append('-march=native')

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'gram_omp',
        sources=['gram_omp.cpp'],
        include_dirs=[get_include.pybind11(), get_include.numpy()],
        language='c++',
    ),
]

setup(
    name='gram_omp',
    version='0.1.0',
    description='OpenMP-accelerated Gram matrix (X X^T, X Y^T) via pybind11',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
