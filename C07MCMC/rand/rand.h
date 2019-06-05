#include <stdlib.h>
#define USE_MATH_DEFINES
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// gauss_box.c
void gen_rand_gauss_box(int num, double* x, double* y);

// gauss_rej.c
void gen_rand_gauss_rej(int num, double* x, double* y);
