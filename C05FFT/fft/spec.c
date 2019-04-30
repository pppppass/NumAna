#include "fft.h"

void solve_spec_3(int size, double* vec, double alpha, double beta, double gamma)
{
    int k = size;
    double* u = vec;
    u[0] /= gamma;
    for (int j = 1; j < 1<<(k-1); j++)
    {
        int x = j, y = j|(1<<(k-1));
        double a = u[x], b = u[y];
        double r = -36.0 * alpha * (double)j * (double)j + gamma, i = 6.0 * beta * (double)j;
        double n2 = r*r + i*i;
        u[x] = (a * r + b * i) / n2;
        u[y] = (b * r - a * i) / n2;
    }
    {
        int j = 1<<(k-1);
        double a = u[j], b = 0.0;
        // cannot really implement the complex version, we take the real component
        double r = -36.0 * alpha * (double)j * (double)j + gamma, i = 6.0 * beta * (double)j;
        double n2 = r*r + i*i;
        u[j] = (a * r + b * i) / n2;
    }
    return ;
}
