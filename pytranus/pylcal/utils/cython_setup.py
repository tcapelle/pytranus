# python cython_setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

ext_modules = [
    Extension("DX",
              ["DX.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
]

setup(
    name={'trololo'},
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)
