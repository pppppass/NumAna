#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

// utils.c
int i_clip(const int val, const int low, const int high);
int index_intv_unif(const int num_node, const double x_low, const double x_high, const double x);
double coor_node_unif(const int num_node, const double x_low, const double x_high, const int index);
int d_lower_bound(const int len, const double* arr, const double val);
double cvrt_ref(const double x_low, const double x_high, const double x);
int index_intv_bis(const int num_node, const double* x_node, const double x);

// newt.c
void calc_newt_arr(int num_node, const double* x_node, const double* y_node, double* coef);
void intp_newt(int num_node, const double* x_node, const double* coef, int num_req, const double* x_req, double* y_req);

// lagr.c
void intp_lagr(int num_node, const double* x_node, const double* y_node, int num_req, const double* x_req, double* y_req);

// lin.c
void intp_lin_unif(int num_node, double x_low, double x_high, const double* y_node, int num_req, const double* x_req, double* y_req);
void intp_lin(int num_node, const double* x_node, const double* y_node, int num_req, const double* x_req, double* y_req);

// cub.c
void intp_cub(int num_node, const double* x_node, const double* y_node, const double* d_y_node, int num_req, const double* x_req, double* y_req);

// spl_cub.c
void calc_spl_cub_d_y_nat(const int num_node, const double* x_node, const double* y_node, double* d_y_node, double* work);
void calc_spl_cub_d_y_coe(int num_node, const double* x_node, const double* y_node, double d_y_low, double d_y_high, double* d_y_node, double* work);
