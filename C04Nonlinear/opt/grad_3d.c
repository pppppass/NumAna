#include "opt.h"

void proj_dir_3d(double *_x, double *_y, double *_z, double a, double b, double c, double d, double u, double v, double w)
{
    if (a * *_x + b * *_y + c * *_z + d <= 0)
        return ;
    else
    {
        double lambda = -(a * *_x + b * *_y + c * *_z + d) / (a * u + b * v + c * w);
        *_x += lambda * u;
        *_y += lambda * v;
        *_z += lambda * w;
    }
    return ;
}

void proj_simp_3d(double *_x, double *_y, double* _z)
{
    proj_dir_3d(_x, _y, _z, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
    proj_dir_3d(_x, _y, _z, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    proj_dir_3d(_x, _y, _z, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    proj_dir_3d(_x, _y, _z, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0);
    return ;
}

//  x <OUT>, y <OUT>, z <OUT>:
//      [0:n]
//  w <OUT>:
//      [0:c]
void proj_para_3d(int num_node, int num_class, double* x, double* y, double* z, double* w)
{
    int n = num_node, c = num_class;
    for (int t = 0; t < n; t++)
        proj_simp_3d(&x[t], &y[t], &z[t]);
    for (int t = 0; t < c; t++)
        w[t] = calc_relu(w[t]);
    return ;
}

//  res_i, res_j, res_k:
//      [0:l]
//  x, y, w_dist:
//      [0:n]
double calc_val_sos_3d_dist(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, const double* x, const double* y, const double* z, const double* w_dist)
{
    int n = num_node, l = num_res;
    double f = 0.0;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_3d(res_i[t], res_j[t], res_k[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w_dist[r] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
        f += f_ij * f_ij / 2.0;
    }
    return f;
}

//  res_i, res_j:
//      [0:l]
//  class_node:
//      [0:n]
//  x, y, z:
//      [0:n]
//  w:
//      [0:c]
//  g_x <OUT>, g_y <OUT>, g_z <OUT>:
//      [0:n]
//  g_w <OUT>:
//      [0:c]
void calc_grad_sos_3d(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, int num_class, const int* class_node, const double* x, const double* y, const double* z, const double* w, double* g_x, double* g_y, double* g_z, double* g_w)
{
    int n = num_node, l = num_res, c = num_class;
    for (int r = 0; r < n; r++)
    {
        g_x[r] = 0.0;
        g_y[r] = 0.0;
        g_z[r] = 0.0;
    }
    for (int r = 0; r < c; r++)
        g_w[r] = 0.0;
    for (int t = 0; t < l; t++)
    {
        double f_ij = calc_int_poly_3d(res_i[t], res_j[t], res_k[t]);
        for (int r = 0; r < n; r++)
            f_ij -= w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_d_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
            g_x[r] += f_ij * d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_d_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
            g_y[r] += f_ij * d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -w[class_node[r]] * calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_d_poly(z[r], res_k[t]);
            g_z[r] += f_ij * d_f_ij;
        }
        for (int r = 0; r < n; r++)
        {
            double d_f_ij = -calc_poly(x[r], res_i[t]) * calc_poly(y[r], res_j[t]) * calc_poly(z[r], res_k[t]);
            g_w[class_node[r]] += f_ij * d_f_ij;
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
//      [0:c]: w
//      [c:n+c]: g_x
//          [0:n]: x_temp
//      [n+c:2*n+c]: g_y
//          [0:n]: y_temp
//      [2*n+c:3*n+c]: g_z
//          [0:n]: z_temp
//      [3*n+c:3*n+2*c]: g_w
//          [0:c]: (int*)w_ctr
//          [0:c]: w_temp
//      [3*n+2*c:4*n+2*c]: x_old
//      [4*n+2*c:5*n+2*c]: y_old
//      [5*n+2*c:6*n+2*c]: y_old
//      [6*n+2*c:6*n+3*c]: w_old
void opt_gauss_sos_3d_grad_nest(int num_node, int num_res, const int* res_i, const int* res_j, const int* res_k, int num_class, const int* class_node, double* x, double* y, double* z, double* w_dist, double eta, int num_iter, double* work)
{
    int n = num_node, l = num_res, c = num_class;
    double* w = work, * g_x = work+c, * g_y = work+(n+c), * g_z = work+(2*n+c), * g_w = work+(3*n+c), * x_old = work+(3*n+2*c), * y_old = work+(4*n+2*c), * z_old = work+(5*n+2*c), * w_old = work+(6*n+2*c);
    double* x_temp = g_x, * y_temp = g_y, * z_temp = g_z, * w_temp = g_w;
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

    proj_para_3d(n, c, x, y, z, w);
    cblas_dcopy(n, x, 1, x_old, 1);
    cblas_dcopy(n, y, 1, y_old, 1);
    cblas_dcopy(n, z, 1, z_old, 1);
    cblas_dcopy(c, w, 1, w_old, 1);
    
    for (int t = 0; t < num_iter; t++)
    {
        cblas_dcopy(n, x, 1, x_temp, 1);
        cblas_dcopy(n, y, 1, y_temp, 1);
        cblas_dcopy(n, z, 1, z_temp, 1);
        cblas_dcopy(c, w, 1, w_temp, 1);

        cblas_dscal(n, (2.0*(double)t + 1.0) / ((double)t + 2.0), x, 1);
        cblas_dscal(n, (2.0*(double)t + 1.0) / ((double)t + 2.0), y, 1);
        cblas_dscal(n, (2.0*(double)t + 1.0) / ((double)t + 2.0), z, 1);
        cblas_dscal(c, (2.0*(double)t + 1.0) / ((double)t + 2.0), w, 1);

        cblas_daxpy(n, -((double)t - 1.0) / ((double)t + 2.0), x_old, 1, x, 1);
        cblas_daxpy(n, -((double)t - 1.0) / ((double)t + 2.0), y_old, 1, y, 1);
        cblas_daxpy(n, -((double)t - 1.0) / ((double)t + 2.0), z_old, 1, z, 1);
        cblas_daxpy(c, -((double)t - 1.0) / ((double)t + 2.0), w_old, 1, w, 1);

        cblas_dcopy(n, x_temp, 1, x_old, 1);
        cblas_dcopy(n, y_temp, 1, y_old, 1);
        cblas_dcopy(n, z_temp, 1, z_old, 1);
        cblas_dcopy(c, w_temp, 1, w_old, 1);
        // x_temp, y_temp, w_temp finished

        calc_grad_sos_3d(n, l, res_i, res_j, res_k, c, class_node, x, y, z, w, g_x, g_y, g_z, g_w);
        
        cblas_daxpy(n, -eta, g_x, 1, x, 1);
        cblas_daxpy(n, -eta, g_y, 1, y, 1);
        cblas_daxpy(n, -eta, g_z, 1, z, 1);
        cblas_daxpy(c, -eta, g_w, 1, w, 1);

        proj_para_3d(n, c, x, y, z, w);
    }

    for (int t = 0; t < n; t++)
        w_dist[t] = w[class_node[t]];

    return ;
}
