#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

void intp_newton(const int num_node, const double* x_node, const double* coef, const int num_req, const double* x_req, double* y_req);
void calc_newton_array(int num_node, const double* x_node, const double* y_node, double* coef);
