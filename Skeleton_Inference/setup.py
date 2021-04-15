try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'external.libkdtree.pykdtree.kdtree',
    sources=[
        'external/libkdtree/pykdtree/kdtree.c',
        'external/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'external.libmcubes.mcubes',
    sources=[
        'external/libmcubes/mcubes.pyx',
        'external/libmcubes/pywrapper.cpp',
        'external/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'external.libsimplify.simplify_mesh',
    sources=[
        'external/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    simplify_mesh_module,
 ]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
