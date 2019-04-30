#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// utils.c
void trans_trans(int size, double* vec);

// dft.c
void trans_dft(int size, double* vec, double* work);

// fft.c
void trans_fft(int size, double* vec);

// ifft.c
void trans_ifft(int size, double* vec);

// diff.c
void solve_diff_3(int size, double* vec, double alpha, double beta, double gamma);

// spec.c
void solve_spec_3(int size, double* vec, double alpha, double beta, double gamma);
