#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// mid.c
double quad_mid_unif(int num_node, double x_low, double x_high, const double* y_node);

// trap.c
double quad_trap_unif(int num_node, double x_low, double x_high, const double* y_node);

// simp.c
double quad_simp_unif(int num_node, double x_low, double x_high, const double* y_node);

// romb.c
double quad_romb_unif(int num_node, int num_order, double x_low, double x_high, const double* y_node, double* work);

// lagu.c
// void calc_lagu_zero(int order, double* zero, double* work);
// double quad_lagu(int num_node, const double* x_node, const double* y_node);
void calc_lagu_para(int order, double* zero, double* weight);

// lege.c
// void calc_lege_zero(int order, double x_low, double x_high, double* zero, double* work);
// double quad_lege(int num_node, double x_low, double x_high, const double* x_node, const double* y_node);
void calc_lege_para(int order, double x_low, double x_high, double* zero, double* weight);
