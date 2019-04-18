#include "opt.h"

//  res_i, res_j:
//      [0:l]
//  x, y, w_dist:
//      [0:n]
//  e <OUT>:
//      [0:l]
void calc_func_2d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const double* x, const double* y, const double* w_dist, double* e)
{
    int n = num_node, l = num_res;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_2d(res_i[t], res_j[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w_dist[r] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
        e[t] = f_ij;
    }
    return ;
}

//  res_i, res_j:
//      [0:l]
//  class_node:
//      [0:n]
//  x, y:
//      [0:n]
//  w:
//      [0:c]
//  e <OUT>:
//      [0:l]
void calc_func_2d(int num_node, int num_res, const int* res_i, const int* res_j, const int* class_node, const double* x, const double* y, const double* w, double* e)
{
    int n = num_node, l = num_res;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_2d(res_i[t], res_j[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
        e[t] = f_ij;
    }
    return ;
}

//  res_i, res_j:
//      [0:l]
//  class_node:
//      [0:n]
//  x, y:
//      [0:n]
//  w:
//      [0:c]
//  _j <OUT>:
//      [0:l*(2*n+c)]:
//          [0:l, 0:2*n+c]: j
void calc_jac_2d(int num_node, int num_res, const int* res_i, const int* res_j, int num_class, const int* class_node, const double* x, const double* y, const double* w, double* _j)
{
    int n = num_node, l = num_res, c = num_class;
    double (* j)[2*n+c] = (void*)_j;
    for (int t = 0; t < l; t++)
    {
        for (int r = 0; r < 2*n+c; r++)
            j[t][r] = 0.0;
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_d_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
            j[t][r] = d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_d_poly(y[r], res_j[t]);
            j[t][r+n] = d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
            j[t][class_node[r]+2*n] += d_f_ij;
        }
    }
    return ;
}

//  res_i, res_j:
//      [0:l]
//  class_node:
//      [0:n]
//  x <OUT>, y <OUT>, w_dist <OUT>:
//      [0:n]
//  work <TEMP>:
//      [0:2*n+c]: p
//          [0:n]: x_con
//          [n:2*n]: y_con
//          [2*n:2*n+c]: w_con
//      [2*n+c:2*(2*n+c)]: g
//      [2*(2*n+c):(l+2)*(2*n+c)]: _j
//          [0:l, 0:2*n+c]: j
//      [(l+2)*(2*n+c):(l+3)*(2*n+c)]: (int*)q
//          [0:c]: (int*)w_ctr
void opt_gauss_2d_newt(int num_node, int num_res, const int* res_i, const int* res_j, int num_class, const int* class_node, double* x, double* y, double* w_dist, double eta, int num_iter, double* work)
{
    int n = num_node, l = num_res, c = num_class;
    double* p = work, * g = work+(2*n+c), * _j = work+2*(2*n+c);
    double* x_con = p, * y_con = p+n, * w_con = p+(2*n);
    int* q = (int*)(work+((l+2)*(2*n+c)));
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
    
    for (int t = 0; t < num_iter; t++)
    {

        calc_func_2d(n, l, res_i, res_j, class_node, x_con, y_con, w_con, g);
        
        calc_jac_2d(n, l, res_i, res_j, c, class_node, x_con, y_con, w_con, _j);

        // implicitly assume l == 2*n+c
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2*n+c, 1, _j, 2*n+c, q, g, 1);

        cblas_daxpy(2*n+c, -eta, g, 1, p, 1);
    }

    cblas_dcopy(n, x_con, 1, x, 1);
    cblas_dcopy(n, y_con, 1, y, 1);
    for (int t = 0; t < n; t++)
        w_dist[t] = w_con[class_node[t]];

    return ;
}
