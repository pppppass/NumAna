#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// eval.c
double calc_lor_u(double x, double y, double z, double sigma, double rho, double beta);
double calc_lor_v(double x, double y, double z, double sigma, double rho, double beta);
double calc_lor_w(double x, double y, double z, double sigma, double rho, double beta);

// ode1.c
void intg_ode1_lor(int num_step, double step, double x0, double y0, double z0, double sigma, double rho, double beta, double* x, double* y, double* z);

// ode4.c
void intg_ode4_lor(int num_step, double step, double x0, double y0, double z0, double sigma, double rho, double beta, double* x, double* y, double* z);
