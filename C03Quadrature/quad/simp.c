#include "quad.h"

//  y_node:
//      [0:2*n+1]
double quad_simp_unif(int num_node, double x_low, double x_high, const double* y_node)
{
    int n = num_node;
    double h = (x_high - x_low) / n;
    double r = 0.0;
    r += (y_node[0] + y_node[2*n]) / 6.0;
    for (int i = 0; i < n-1; i++)
        r += (2.0 / 3.0) * y_node[2*i+1] + y_node[2*i+2] / 3.0;
    r += (2.0 / 3.0) * y_node[2*n-1];
    r *= h;
    return r;
}
