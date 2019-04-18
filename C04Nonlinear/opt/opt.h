#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// utils.c
double calc_relu(double v);
double calc_poly(double x, int k);
double calc_d_poly(double x, int k);
double calc_d_d_poly(double x, int k);
double calc_int_poly_2d(int i, int j);
double calc_int_poly_3d(int i, int j, int k);

// grad_2d.c
double calc_val_sos_2d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const double* x, const double* y, const double* w_dist);
void opt_gauss_sos_2d_grad_nest(int num_node, int num_res, const int* res_i, const int* res_j, int num_class, const int* class_node, double* x, double* y, double* w_dist, double eta, int num_iter, double* work);

// newt_2d.c
void calc_func_2d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const double* x, const double* y, const double* w_dist, double* e);
void opt_gauss_2d_newt(int num_node, int num_res, const int* res_i, const int* res_j, int num_class, const int* class_node, double* x, double* y, double* w_dist, double eta, int num_iter, double* work);

// grad_3d.c
double calc_val_sos_3d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, const double* x, const double* y, const double* z, const double* w_dist);
void opt_gauss_sos_3d_grad_nest(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, int num_class, const int* class_node, double* x, double* y, double* z, double* w_dist, double eta, int num_iter, double* work);

// newt_3d.c
void calc_func_3d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, const double* x, const double* y, const double* z, const double* w_dist, double* e);
void opt_gauss_3d_newt(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, int num_class, const int* class_node, double* x, double* y, double* z, double* w_dist, double eta, int num_iter, double* work);
