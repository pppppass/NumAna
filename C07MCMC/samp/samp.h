#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

// utils.c
// void iswap(int* a, int* b);
void pop_and_push_2d(int size, int x, int y, int* q_c, int* q_coor, int ctr_c[10], int* c_stack_, int inc);
void pop_and_push_3d(int size, int x, int y, int z, int* q_c, int* q_coor, int ctr_c[10], int* c_stack_, int inc);

// sing_2d.c
void samp_ising_single_2d(int size, double temp, double bias, int iter, VSLStreamStatePtr s, int size_buf, int* work_int, double* work_dbl, int* site);
void driver_samp_ising_single_2d(int size, double temp, double bias, int iter, int rep, int size_buf, int seed, int* sites);

// metr_2d.c
void samp_ising_2d(int size, double temp, double bias, int iter, int start, VSLStreamStatePtr s, int size_buf, int* work_int, double* work_dbl, double* m, double* ma, double* u, double* c);
void driver_samp_ising_2d(int size, double temp, double bias, int iter, int start, int rep, int size_buf, int seed, double* m1, double* m2, double* ma1, double* ma2, double* u1, double* u2, double* c1, double* c2);

// kin_2d.c
void samp_ising_kin_2d(int size, double temp, double bias, int iter, int start, VSLStreamStatePtr s, int size_buf, int* work_int, double* work_dbl, double* m, double* ma, double* u, double* c);
void driver_samp_ising_kin_2d(int size, double temp, double bias, int iter, int start, int rep, int size_buf, int seed, double* m1, double* m2, double* ma1, double* ma2, double* u1, double* u2, double* c1, double* c2);

// kin_3d.c
void samp_ising_kin_3d(int size, double temp, double bias, int iter, int start, VSLStreamStatePtr s, int size_buf, int* work_int, double* work_dbl, double* m, double* ma, double* u, double* c);
void driver_samp_ising_kin_3d(int size, double temp, double bias, int iter, int start, int rep, int size_buf, int seed, double* m1, double* m2, double* ma1, double* ma2, double* u1, double* u2, double* c1, double* c2);
