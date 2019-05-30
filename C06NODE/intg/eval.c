#include "intg.h"

double calc_lor_u(double x, double y, double z, double sigma, double rho, double beta)
{
    double u = sigma * (y - x);
    return u;
}

double calc_lor_v(double x, double y, double z, double sigma, double rho, double beta)
{
    double v = rho * x - y - x * z;
    return v;
}

double calc_lor_w(double x, double y, double z, double sigma, double rho, double beta)
{
    double w = x * y - beta * z;
    return w;
}
