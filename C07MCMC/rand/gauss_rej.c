#include "rand.h"

void gen_rand_gauss_rej(int num, double* x, double* y)
{
    int n = num;
    for (int i = 0; i < n; i++)
    {
        double u = drand48(), v = drand48();
        u = 2.0 * u - 1.0, v = 2.0 * v - 1.0;
        double r_2 = u*u + v*v;
        if (r_2 >= 1.0)
        {
            i--;
            continue;
        }
        double x_t = u * sqrt(-2.0 * log(r_2) / r_2), y_t = v * sqrt(-2.0 * log(r_2) / r_2);
        x[i] = x_t, y[i] = y_t;
    }
    return ;
}
