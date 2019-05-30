#include "intg.h"

void intg_ode4_lor(int num_step, double step, double x0, double y0, double z0, double sigma, double rho, double beta, double* x, double* y, double* z)
{
    int n = num_step;
    double h = step;
    double xt = x0, yt = y0, zt = z0;
    for (int i = 0; i < n; i++)
    {
        x[i] = xt, y[i] = yt, z[i] = zt;
        double u1 = calc_lor_u(xt, yt, zt, sigma, rho, beta), v1 = calc_lor_v(xt, yt, zt, sigma, rho, beta), w1 = calc_lor_w(xt, yt, zt, sigma, rho, beta);
        double u2 = calc_lor_u(xt + h / 2.0 * u1, yt + h / 2.0 * v1, zt + h / 2.0 * w1, sigma, rho, beta), v2 = calc_lor_v(xt + h / 2.0 * u1, yt + h / 2.0 * v1, zt + h / 2.0 * w1, sigma, rho, beta), w2 = calc_lor_w(xt + h / 2.0 * u1, yt + h / 2.0 * v1, zt + h / 2.0 * w1, sigma, rho, beta);
        double u3 = calc_lor_u(xt + h / 2.0 * u2, yt + h / 2.0 * v2, zt + h / 2.0 * w2, sigma, rho, beta), v3 = calc_lor_v(xt + h / 2.0 * u2, yt + h / 2.0 * v2, zt + h / 2.0 * w2, sigma, rho, beta), w3 = calc_lor_w(xt + h / 2.0 * u2, yt + h / 2.0 * v2, zt + h / 2.0 * w2, sigma, rho, beta);
        double u4 = calc_lor_u(xt + h * u3, yt + h * v3, zt + h * w3, sigma, rho, beta), v4 = calc_lor_v(xt + h * u3, yt + h * v3, zt + h * w3, sigma, rho, beta), w4 = calc_lor_w(xt + h * u3, yt + h * v3, zt + h * w3, sigma, rho, beta);
        xt += h / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4), yt += h / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4), zt += h / 6.0 * (w1 + 2.0 * w2 + 2.0 * w3 + w4);
    }
    return ;
}
