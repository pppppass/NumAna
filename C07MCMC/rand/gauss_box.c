#include "rand.h"

void gen_rand_gauss_box(int num, double* x, double* y)
{
    int n = num;
    for (int i = 0; i < n; i++)
    {
        double u = drand48(), v = drand48();
        double x_t = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v), y_t = sqrt(-2.0 * log(u)) * sin(2.0 * M_PI * v);
        x[i] = x_t, y[i] = y_t;
    }
    return ;
}
