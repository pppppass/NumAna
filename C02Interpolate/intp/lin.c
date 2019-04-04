#include "intp.h"

double intp_lin_ref(double y_0, double y_1, double x)
{
    return 
          y_0 * (1.0 - x)
        + y_1 * x;
}

//  y_node:
//      [0:n+1]
//  x_req, y_req <OUT>:
//      [0:m]
void intp_lin_unif(int num_node, double x_low, double x_high, const double* y_node, int num_req, const double* x_req, double* y_req)
{
    int n = num_node, m = num_req;
    for (int j = 0; j < m; j++)
    {
        int i = index_intv_unif(n, x_low, x_high, x_req[j]);
        double
            x_low_i = coor_node_unif(n, x_low, x_high, i),
            x_high_i = coor_node_unif(n, x_low, x_high, i+1);
        double x_ref = cvrt_ref(x_low_i, x_high_i, x_req[j]);
        y_req[j] = intp_lin_ref(y_node[i], y_node[i+1], x_ref);
    }
    return ;
}

//  x_node, y_node:
//      [0:n+1]:
//          x_node: strictly increasing
//  x_req, y_req <OUT>:
//      [0:m]
void intp_lin(int num_node, const double* x_node, const double* y_node, int num_req, const double* x_req, double* y_req)
{
    int n = num_node, m = num_req;
    for (int j = 0; j < m; j++)
    {
        int i = index_intv_bis(n, x_node, x_req[j]);
        double x_ref = cvrt_ref(x_node[i], x_node[i+1], x_req[j]);
        y_req[j] = intp_lin_ref(y_node[i], y_node[i+1], x_ref);
    }
    return ;
}
