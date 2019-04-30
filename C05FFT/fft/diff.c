#include "fft.h"

void solve_diff_3(int size, double* vec, double alpha, double beta, double gamma)
{
    int k = size;
    double* u = vec;
    u[0] /= (alpha + beta + gamma);
    u[1<<(k-1)] /= (-alpha + beta - gamma);
    for (int j = 1; j < 1<<(k-1); j++)
    {
        int x = j, y = j|(1<<(k-1));
        double a = u[x], b = u[y];
        double theta = 2.0 * M_PI * (double)j / (double)(1<<k);
        double r = beta + (alpha + gamma) * cos(theta), i = (alpha - gamma) * sin(theta);
        double n2 = r*r + i*i;
        u[x] = (a * r + b * i) / n2;
        u[y] = (b * r - a * i) / n2;
    }
    return ;
}
