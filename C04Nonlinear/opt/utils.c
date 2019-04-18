#include "opt.h"

double calc_relu(double v)
{
    if (v < 0.0)
        return 0.0;
    else
        return v;
}

double calc_poly(double x, int k)
{
    if (k <= 0)
        return 1.0;
    else
        return pow(x, (double)k);
}

double calc_d_poly(double x, int k)
{
    if (k <= 0)
        return 0.0;
    else if (k == 1)
        return 1.0;
    else
        return (double)k * pow(x, (double)(k-1));
}

double calc_d_d_poly(double x, int k)
{
    if (k <= 1)
        return 0.0;
    else if (k == 2)
        return 2.0;
    else
        return (double)k * (double)(k-1) * pow(x, (double)(k-2));
}

double calc_int_poly_2d(int i, int j)
{
    double log_c = lgamma((double)(i+1)) + lgamma((double)(j+1)) - lgamma((double)(i+j+3));
    return exp(log_c);
}

double calc_int_poly_3d(int i, int j, int k)
{
    double log_c = lgamma((double)(i+1)) + lgamma((double)(j+1)) + lgamma((double)(k+1)) - lgamma((double)(i+j+k+4));
    return exp(log_c);
}
