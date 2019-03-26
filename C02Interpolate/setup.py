from distutils.core import setup, Extension
import numpy

intp = Extension("intp", sources=["intp/newt.c", "intp/lagr.c", "intp/lin.c", "intp/cub.c", "intp/spl_cub.c", "intp/utils.c", "intp/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m"], extra_compile_args=["-march=skylake"])

setup(ext_modules=[intp])
