from distutils.core import setup, Extension
import numpy

quad = Extension("quad", sources=["quad/mid.c", "quad/trap.c", "quad/simp.c", "quad/romb.c", "quad/lagu.c", "quad/lege.c", "quad/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m"], extra_compile_args=["-march=native"])

setup(ext_modules=[quad])
