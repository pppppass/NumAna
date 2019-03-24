from distutils.core import setup, Extension
import numpy

intp = Extension("intp", sources=["intp/algos.c", "intp/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m"], extra_compile_args=["-march=skylake"])

setup(ext_modules=[intp])
