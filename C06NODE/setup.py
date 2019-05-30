from distutils.core import setup, Extension
import numpy

intg = Extension("intg", sources=["intg/eval.c", "intg/ode1.c", "intg/ode4.c", "intg/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m"], extra_compile_args=["-march=native"])

setup(ext_modules=[intg])
