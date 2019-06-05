from distutils.core import setup, Extension
import numpy

samp = Extension("samp", sources=["samp/sing_2d.c", "samp/metr_2d.c", "samp/kin_2d.c", "samp/kin_3d.c", "samp/utils.c", "samp/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m", "mkl_rt"], extra_compile_args=["-march=native", "-fopenmp"], extra_link_args=["-fopenmp"])

setup(ext_modules=[samp])

rand = Extension("rand", sources=["rand/gauss_box.c", "rand/gauss_rej.c", "rand/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m"], extra_compile_args=["-march=native"])

setup(ext_modules=[rand])
