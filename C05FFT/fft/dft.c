#include "fft.h"

//  u <OUT>:
//      [0:2^k]
//  w <TEMP>:
//      [0:2^k]
void trans_dft(int size, double* vec, double* work)
{
    int k = size;
    double* u = vec, * w = work;
    int n = 1<<k;
    for (int i = 0; i <= n/2; i++)
    {
        double v = 0.0;
        for (int j = 0; j < n; j++)
            v += u[j] * cos(2.0 * M_PI * (double)i * (double)j / (double)n);
        w[i] = v;
    }
    for (int i = 1; i < n/2; i++)
    {
        double v = 0.0;
        for (int j = 0; j < n; j++)
            v += u[j] * sin(2.0 * M_PI * (double)i * (double)j / (double)n);
        w[i+n/2] = -v;
    }
    cblas_dcopy(n, w, 1, u, 1);
    return ;
}
