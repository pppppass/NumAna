#include "intg.h"

void intg_ode1_lor(int num_step, double step, double x0, double y0, double z0, double sigma, double rho, double beta, double* x, double* y, double* z)
{
    int n = num_step;
    double h = step;
    double xt = x0, yt = y0, zt = z0;
    for (int i = 0; i < n; i++)
    {
        x[i] = xt, y[i] = yt, z[i] = zt;
        double u = calc_lor_u(xt, yt, zt, sigma, rho, beta), v = calc_lor_v(xt, yt, zt, sigma, rho, beta), w = calc_lor_w(xt, yt, zt, sigma, rho, beta);
        xt += h * u, yt += h * v, zt += h * w;
    }
    return ;
}
