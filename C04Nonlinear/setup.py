from distutils.core import setup, Extension
import numpy

opt = Extension("opt", sources=["opt/grad_2d.c", "opt/newt_2d.c", "opt/grad_3d.c", "opt/newt_3d.c", "opt/utils.c", "opt/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m"], extra_compile_args=["-march=native"])

setup(ext_modules=[opt])
