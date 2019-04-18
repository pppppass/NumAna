#include "opt.h"

void proj_dir_2d(double *_x, double *_y, double a, double b, double c, double u, double v)
{
    if (a * *_x + b * *_y + c <= 0)
        return ;
    else
    {
        double lambda = -(a * *_x + b * *_y + c) / (a * u + b * v);
        *_x += lambda * u;
        *_y += lambda * v;
    }
    return ;
}

void proj_simp_2d(double *_x, double *_y)
{
    // proj_dir_2d(_x, _y, 0.0, -1.0, 0.0, 0.0, 1.0);
    // proj_dir_2d(_x, _y, -1.0, 0.0, 0.0, 2.0, -1.0);
    // proj_dir_2d(_x, _y, 1.0, 1.0, -1.0, 1.0, 1.0);
    proj_dir_2d(_x, _y, 0.0, -1.0, 0.0, 0.0, 1.0);
    proj_dir_2d(_x, _y, -1.0, 0.0, 0.0, 1.0, 0.0);
    proj_dir_2d(_x, _y, 1.0, 1.0, -1.0, 1.0, 1.0);
    return ;
}

//  x <OUT>, y <OUT>:
//      [0:n]
//  w <OUT>:
//      [0:c]
void proj_para_2d(int num_node, int num_class, double* x, double* y, double* w)
{
    int n = num_node, c = num_class;
    for (int t = 0; t < n; t++)
        proj_simp_2d(&x[t], &y[t]);
    for (int t = 0; t < c; t++)
        w[t] = calc_relu(w[t]);
    return ;
}

//  res_i, res_j:
//      [0:l]
//  x, y, w_dist:
//      [0:n]
double calc_val_sos_2d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const double* x, const double* y, const double* w_dist)
{
    int n = num_node, l = num_res;
    double f = 0.0;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_2d(res_i[t], res_j[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w_dist[r] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
        f += f_ij * f_ij / 2.0;
    }
    return f;
}

//  res_i, res_j:
//      [0:l]
//  class_node:
//      [0:n]
//  x, y:
//      [0:n]
//  w:
//      [0:c]
//  g_x <OUT>, g_y <OUT>:
//      [0:n]
//  g_w <OUT>:
//      [0:c]
void calc_grad_sos_2d(int num_node, int num_res, const int* res_i, const int* res_j, int num_class, const int* class_node, const double* x, const double* y, const double* w, double* g_x, double* g_y, double* g_w)
{
    int n = num_node, l = num_res, c = num_class;
    for (int r = 0; r < n; r++)
    {
        g_x[r] = 0.0;
        g_y[r] = 0.0;
    }
    for (int r = 0; r < c; r++)
        g_w[r] = 0.0;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_2d(res_i[t], res_j[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_d_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
            g_x[r] += f_ij * d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_d_poly(y[r], res_j[t]);
            g_y[r] += f_ij * d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]);
            g_w[class_node[r]] += f_ij * d_f_ij;
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
//      [0:c]: w
//      [c:n+c]: g_x
//          [0:n]: x_temp
//      [n+c:2*n+c]: g_y
//          [0:n]: y_temp
//      [2*n+c:2*n+2*c]: g_w
//          [0:c]: (int*)w_ctr
//          [0:c]: w_temp
//      [2*n+2*c:3*n+2*c]: x_old
//      [3*n+2*c:4*n+2*c]: y_old
//      [4*n+2*c:4*n+3*c]: w_old
void opt_gauss_sos_2d_grad_nest(int num_node, int num_res, const int* res_i, const int* res_j, int num_class, const int* class_node, double* x, double* y, double* w_dist, double eta, int num_iter, double* work)
{
    int n = num_node, l = num_res, c = num_class;
    double* w = work, * g_x = work+c, * g_y = work+(n+c), * g_w = work+(2*n+c), * x_old = work+(2*n+2*c), * y_old = work+(3*n+2*c), * w_old = work+(4*n+2*c);
    double* x_temp = g_x, * y_temp = g_y, * w_temp = g_w;
    int* w_ctr = (int*)g_w;
    
    // take average from w_dist to w
    for (int t = 0; t < c; t++)
    {
        w[t] = 0.0;
        w_ctr[t] = 0;
    }
    for (int t = 0; t < n; t++)
    {
        w[class_node[t]] += w_dist[t];
        w_ctr[class_node[t]] += 1;
    }
    for (int t = 0; t < c; t++)
        w[t] /= (double)w_ctr[t];
    // w_ctr finished

    proj_para_2d(n, c, x, y, w);
    cblas_dcopy(n, x, 1, x_old, 1);
    cblas_dcopy(n, y, 1, y_old, 1);
    cblas_dcopy(c, w, 1, w_old, 1);
    
    for (int t = 0; t < num_iter; t++)
    {
        cblas_dcopy(n, x, 1, x_temp, 1);
        cblas_dcopy(n, y, 1, y_temp, 1);
        cblas_dcopy(c, w, 1, w_temp, 1);

        cblas_dscal(n, (2.0*(double)t + 1.0) / ((double)t + 2.0), x, 1);
        cblas_dscal(n, (2.0*(double)t + 1.0) / ((double)t + 2.0), y, 1);
        cblas_dscal(c, (2.0*(double)t + 1.0) / ((double)t + 2.0), w, 1);

        cblas_daxpy(n, -((double)t - 1.0) / ((double)t + 2.0), x_old, 1, x, 1);
        cblas_daxpy(n, -((double)t - 1.0) / ((double)t + 2.0), y_old, 1, y, 1);
        cblas_daxpy(c, -((double)t - 1.0) / ((double)t + 2.0), w_old, 1, w, 1);

        cblas_dcopy(n, x_temp, 1, x_old, 1);
        cblas_dcopy(n, y_temp, 1, y_old, 1);
        cblas_dcopy(c, w_temp, 1, w_old, 1);
        // x_temp, y_temp, w_temp finished

        calc_grad_sos_2d(n, l, res_i, res_j, c, class_node, x, y, w, g_x, g_y, g_w);
        
        cblas_daxpy(n, -eta, g_x, 1, x, 1);
        cblas_daxpy(n, -eta, g_y, 1, y, 1);
        cblas_daxpy(c, -eta, g_w, 1, w, 1);

        proj_para_2d(n, c, x, y, w);
    }

    for (int t = 0; t < n; t++)
        w_dist[t] = w[class_node[t]];

    return ;
}
