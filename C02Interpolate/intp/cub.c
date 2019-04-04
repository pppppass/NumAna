#include "intp.h"

double intp_cub_ref(double y_0, double y_1, double d_y_0, double d_y_1, double x)
{
    return 
          y_0 * (x - 1.0) * (x - 1.0) * (2.0 * x + 1.0)
        + y_1 * x * x * (3.0 - 2.0 * x)
        + d_y_0 * (x - 1.0) * (x - 1.0) * x
        + d_y_1 * x * x * (x - 1.0);
}

//  x_node, y_node, d_y_node:
//      [0:n+1]
//  x_req, y_req <OUT>:
//      [0:m]
void intp_cub(int num_node, const double* x_node, const double* y_node, const double* d_y_node, int num_req, const double* x_req, double* y_req)
{
    int n = num_node, m = num_req;
    for (int j = 0; j < m; j++)
    {
        int i = index_intv_bis(n, x_node, x_req[j]);
        double x_ref = cvrt_ref(x_node[i], x_node[i+1], x_req[j]);
        double h = x_node[i+1] - x_node[i];
        y_req[j] = intp_cub_ref(y_node[i], y_node[i+1], d_y_node[i] * h, d_y_node[i+1] * h, x_ref);
    }
    return ;
}
