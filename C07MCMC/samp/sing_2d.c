#include "samp.h"

void samp_ising_single_2d(int size, double temp, double bias, int iter, VSLStreamStatePtr s, int size_buf, int* work_int, double* work_dbl, int* site)
{
    int n = size;
    double t = temp, h = bias;

    int* q = site;
    int(* q_)[n] = (void*)q;
    int* w_int = work_int;
    double* w_dbl = work_dbl;
    int ctr_w = size_buf;
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, s, n*n, q, 0, 2);

    for (int i = 0; i < n*n; i++)
        q[i] = 2 * q[i] - 1;

    for (int i = 0; i < iter; i++)
    {
        if (ctr_w >= size_buf)
        {
            viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, s, size_buf, w_int, 0, n*n);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, s, size_buf, w_dbl, 0.0, 1.0);
            ctr_w = 0;
        }
        int x = w_int[ctr_w] / n, y = w_int[ctr_w] % n;
        int 
            x_w = (x + n - 1) % n, x_e = (x + 1) % n,
            y_s = (y + n - 1) % n, y_n = (y + 1) % n;
        int d = -q_[x][y] * (q_[x_w][y] + q_[x_e][y] + q_[x][y_s] + q_[x][y_n]);
        double delta = -2.0 * d + 2.0 * q_[x][y] * h;
        if (w_dbl[ctr_w] < exp(-delta / t))
            q_[x][y] = -q_[x][y];
        ctr_w++;
    }

    return ;
}

void driver_samp_ising_single_2d(int size, double temp, double bias, int iter, int rep, int size_buf, int seed, int* sites)
{
    int n = size, m = rep;
    double t = temp, h = bias;
    int* q = sites;
#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        VSLStreamStatePtr s;
        vslNewStream(&s, VSL_BRNG_MCG31, seed+rank);
        int* w_int = malloc(size_buf * sizeof(int));
        double* w_dbl = malloc(size_buf * sizeof(double));
#pragma omp for schedule(static)
        for (int i = 0; i < m; i++)
        {
            samp_ising_single_2d(n, t, h, iter, s, size_buf, w_int, w_dbl, q + i*(n*n));
        }
        free(w_int), free(w_dbl);
        vslDeleteStream(&s);
    }
    return ;
}
