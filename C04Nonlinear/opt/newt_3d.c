#include "opt.h"

//  res_i, res_j, res_k:
//      [0:l]
//  x, y, z, w_dist:
//      [0:n]
//  e <OUT>:
//      [0:l]
void calc_func_3d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, const double* x, const double* y, const double* z, const double* w_dist, double* e)
{
    int n = num_node, l = num_res;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_3d(res_i[t], res_j[t], res_k[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w_dist[r] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
        e[t] = f_ij;
    }
    return ;
}

//  res_i, res_j, res_k:
//      [0:l]
//  class_node:
//      [0:n]
//  x, y, z:
//      [0:n]
//  w:
//      [0:c]
//  e <OUT>:
//      [0:l]
void calc_func_3d(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, const int* class_node, const double* x, const double* y, const double* z, const double* w, double* e)
{
    int n = num_node, l = num_res;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_3d(res_i[t], res_j[t], res_k[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
        e[t] = f_ij;
    }
    return ;
}

//  res_i, res_j, res_k:
//      [0:l]
//  class_node:
//      [0:n]
//  x, y, z:
//      [0:n]
//  w:
//      [0:c]
//  _j <OUT>:
//      [0:l*(3*n+c)]:
//          [0:l, 0:3*n+c]: j
void calc_jac_3d(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, int num_class, const int* class_node, const double* x, const double* y, const double* z, const double* w, double* _j)
{
    int n = num_node, l = num_res, c = num_class;
    double (* j)[3*n+c] = (void*)_j;
    for (int t = 0; t < l; t++)
    {
        for (int r = 0; r < 3*n+c; r++)
            j[t][r] = 0.0;
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_d_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
            j[t][r] = d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_d_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
            j[t][r+n] = d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_d_poly(z[r], res_k[t]);
            j[t][r+2*n] = d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
            j[t][class_node[r]+3*n] += d_f_ij;
        }
    }
    return ;
}

//  res_i, res_j, res_k:
//      [0:l]
//  class_node:
//      [0:n]
//  x <OUT>, y <OUT>, z <OUT>, w_dist <OUT>:
//      [0:n]
//  work <TEMP>:
//      [0:3*n+c]: p
//          [0:n]: x_con
//          [n:2*n]: y_con
//          [2*n:3*n]: z_con
//          [3*n:3*n+c]: w_con
//      [3*n+c:2*(3*n+c)]: g
//      [2*(3*n+c):(l+2)*(3*n+c)]: _j
//          [0:l, 0:3*n+c]: j
//      [(l+2)*(3*n+c):(l+3)*(3*n+c)]: (int*)q
//          [0:c]: (int*)w_ctr
void opt_gauss_3d_newt(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, int num_class, const int* class_node, double* x, double* y, double* z, double* w_dist, double eta, int num_iter, double* work)
{
    int n = num_node, l = num_res, c = num_class;
    double* p = work, * g = work+(3*n+c), * _j = work+2*(3*n+c);
    double* x_con = p, * y_con = p+n, * z_con = p+(2*n), * w_con = p+(3*n);
    int* q = (int*)(work+((l+2)*(3*n+c)));
    int* w_ctr = q;
    
    // take average from w_dist to w
    for (int t = 0; t < c; t++)
    {
        w_con[t] = 0.0;
        w_ctr[t] = 0;
    }
    for (int t = 0; t < n; t++)
    {
        w_con[class_node[t]] += w_dist[t];
        w_ctr[class_node[t]] += 1;
    }
    for (int t = 0; t < c; t++)
        w_con[t] /= (double)w_ctr[t];
    // w_ctr finished

    cblas_dcopy(n, x, 1, x_con, 1);
    cblas_dcopy(n, y, 1, y_con, 1);
    cblas_dcopy(n, z, 1, z_con, 1);
    
    for (int t = 0; t < num_iter; t++)
    {

        calc_func_3d(n, l, res_i, res_j, res_k, class_node, x_con, y_con, z_con, w_con, g);
        
        calc_jac_3d(n, l, res_i, res_j, res_k, c, class_node, x_con, y_con, z_con, w_con, _j);

        // implicitly assume l == 3*n+c
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, 3*n+c, 1, _j, 3*n+c, q, g, 1);

        cblas_daxpy(3*n+c, -eta, g, 1, p, 1);
    }

    cblas_dcopy(n, x_con, 1, x, 1);
    cblas_dcopy(n, y_con, 1, y, 1);
    cblas_dcopy(n, z_con, 1, z, 1);
    for (int t = 0; t < n; t++)
        w_dist[t] = w_con[class_node[t]];

    return ;
}
