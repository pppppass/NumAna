#include "quad.h"

//  y_node:
//      [0:n]
double quad_mid_unif(int num_node, double x_low, double x_high, const double* y_node)
{
    int n = num_node;
    double h = (x_high - x_low) / n;
    double r = 0.0;
    for (int i = 0; i < n; i++)
        r += y_node[i];
    r *= h;
    return r;
}
