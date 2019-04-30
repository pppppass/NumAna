from distutils.core import setup, Extension
import numpy

fft = Extension("fft", sources=["fft/dft.c", "fft/fft.c", "fft/ifft.c", "fft/diff.c", "fft/spec.c", "fft/utils.c", "fft/wrappers.c"], include_dirs=[numpy.get_include()], libraries=["m"], extra_compile_args=["-march=native"])

setup(ext_modules=[fft])
