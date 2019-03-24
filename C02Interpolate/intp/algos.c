#include "intp.h"

void intp_newton(const int num_node, const double* x_node, const double* coef, const int num_req, const double* x_req, double* y_req)
{
    int n = num_node, m = num_req;
    const double* c = coef;
    for (int j = 0; j < m; j++)
        y_req[j] = c[n];
    for (int i = n-1; i >= 0; i--)
        for (int j = 0; j < m; j++)
            y_req[j] = (x_req[j] - x_node[i]) * y_req[j] + c[i];
    return ;
}

void calc_newton_array(int num_node, const double* x_node, const double* y_node, double* coef)
{
    int n = num_node;
    double* c = coef;
    c[0] = y_node[0];
    for (int i = 1; i <= n; i++)
    {
        c[i] = y_node[i];
        for (int j = 0; j < i; j++)
            c[i] = (c[i] - c[j]) / (x_node[i] - x_node[j]);
    }
    return ;
}
