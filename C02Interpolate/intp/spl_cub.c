#include "intp.h"

//  x_node, y_node:
//      [0:n+1]
//  d_y_node <OUT>:
//      [0:n+1]:
//  work <TEMP>:
//      [0:n+1]: d[i] = 2.0 
//          [0:n]: D_x[i] = x[i+1] - x[i] 
//      [n+1:2*n+1]: u[i] = $\lambda_i$
//      [2*n+1:3*n+1]: l[i] = $ 1 - \lambda_{ i + 1 } $
//          [0:n]: D_y_D_x[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
void calc_spl_cub_d_y_nat(int num_node, const double* x_node, const double* y_node, double* d_y_node, double* work)
{
    int n = num_node;
    double* r = d_y_node;
    double* d = work, * u = work+(n+1), * l = work+(2*n+1);
    double* D_x = work, * D_y_D_x = work+(2*n+1);
    for (int i = 0; i < n; i++)
    {
        D_x[i] = (x_node[i+1] - x_node[i]);
        D_y_D_x[i] = (y_node[i+1] - y_node[i]) / D_x[i];
    }
    r[0] = 3.0 * D_y_D_x[0];
    r[n] = 3.0 * D_y_D_x[n-1];
    for (int i = 1; i < n; i++)
    {
        u[i] = D_x[i-1] / (D_x[i-1] + D_x[i]);
        r[i] = 3.0 * ((1.0 - u[i]) * D_y_D_x[i-1] + u[i] * D_y_D_x[i]);
    }
    // D_x, D_y_D_x finished
    for (int i = 0; i < n-1; i++)
        l[i] = 1.0 - u[i+1];
    u[0] = 1.0;
    l[n-1] = 1.0;
    for (int i = 0; i <= n; i++)
        d[i] = 2.0;
    // solve in-place: d_y_node[0:n+1] = r[0:n+1]
    LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n+1, 1, l, d, u, r, 1);
    return ;
}

//  x_node, y_node:
//      [0:n+1]
//  d_y_node <OUT>:
//      [0:n+1]
//  work <TEMP>:
//      [0:n-1] : d[i] = 2.0
//          [0:n] : D_x[i] = x[i+1] - x[i]
//      [n:2*n-1] : u[i] = $ \lambda_{ i + 1 } $
//      [2*n-1:3*n-1] : l[i] = $ 1 - \lambda_{ i + 2 } $
//          [0:n] : D_y_D_x[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
void calc_spl_cub_d_y_coe(int num_node, const double* x_node, const double* y_node, double d_y_low, double d_y_high, double* d_y_node, double* work)
{
    int n = num_node;
    d_y_node[0] = d_y_low;
    d_y_node[n] = d_y_high;
    double* r = d_y_node+1;
    double* d = work, * u = work+(n), * l = work+(2*n-1);
    double* D_x = work, * D_y_D_x = work+(2*n-1);
    for (int i = 0; i < n; i++)
    {
        D_x[i] = (x_node[i+1] - x_node[i]);
        D_y_D_x[i] = (y_node[i+1] - y_node[i]) / D_x[i];
    }
    for (int i = 0; i < n-1; i++)
    {
        u[i] = D_x[i] / (D_x[i] + D_x[i+1]);
        r[i] = 3.0 * ((1.0 - u[i]) * D_y_D_x[i] + u[i] * D_y_D_x[i+1]);
    }
    // D_x, D_y_D_x finished
    r[0] -= (1.0 - u[0]) * d_y_low;
    r[n-2] -= u[n-2] * d_y_high;
    for (int i = 0; i < n-2; i++)
        l[i] = 1.0 - u[i+1];
    for (int i = 0; i < n-1; i++)
        d[i] = 2.0;
    // solve in-place: d_y_node[1:n] = r[0:n-1]
    LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n-1, 1, l, d, u, r, 1);
    return ;
}
