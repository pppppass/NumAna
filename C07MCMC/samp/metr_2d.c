#include "samp.h"

void samp_ising_2d(int size, double temp, double bias, int iter, int start, VSLStreamStatePtr s, int size_buf, int* work_int, double* work_dbl, double* m, double* ma, double* u, double* c)
{
    int n = size;
    double t = temp, h = bias;

    int* q = work_int, * w_int = work_int + n*n;
    int(* q_)[n] = (void*)q;
    double* w_dbl = work_dbl;
    int ctr_w = size_buf;
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, s, n*n, q, 0, 2);

    int m_now = 0, h_now = 0;
    double m1 = 0.0, ma1 = 0.0, h1 = 0.0, h2 = 0.0;
    for (int i = 0; i < n*n; i++)
    {
        q[i] = 2 * q[i] - 1;
        m_now += q[i];
    }
    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
        {
            int x_e = (x+1) % n, y_n = (y+1) % n;
            h_now -= q_[x][y] * (q_[x_e][y] + q_[x][y_n]);
        }

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
        {
            m_now -= 2 * q_[x][y];
            h_now -= 2 * d;
            q_[x][y] = -q_[x][y];
        }
        if (i >= start)
            m1 += m_now, ma1 += fabs(m_now), h1 += h_now, h2 += h_now*h_now;
        ctr_w++;
    }

    m1 /= (iter - start), ma1 /= (iter - start), h1 /= (iter - start), h2 /= (iter - start);
    *m = m1 / (n*n), *ma = ma1 / (n*n), *u = h1 / (n*n), *c = (h2 - h1*h1) / (t*t) / (n*n);
    return ;
}

void driver_samp_ising_2d(int size, double temp, double bias, int iter, int start, int rep, int size_buf, int seed, double* m1, double* m2, double* ma1, double* ma2, double* u1, double* u2, double* c1, double* c2)
{
    int n = size, m = rep;
    double t = temp, h = bias;
#pragma omp parallel reduction(+: m1[:1], m2[:1], ma1[:1], ma2[:1], u1[:1], u2[:1], c1[:1], c2[:1])
    {
        int rank = omp_get_thread_num();
        VSLStreamStatePtr s;
        vslNewStream(&s, VSL_BRNG_MCG31, seed+rank);
        int* w_int = malloc((n*n + size_buf) * sizeof(int));
        double* w_dbl = malloc(size_buf * sizeof(double));
#pragma omp for schedule(static)
        for (int i = 0; i < m; i++)
        {
            double m, ma, u, c;
            samp_ising_2d(n, t, h, iter, start, s, size_buf, w_int, w_dbl, &m, &ma, &u, &c);
            m1[0] += m, m2[0] += m*m;
            ma1[0] += ma, ma2[0] += ma*ma;
            u1[0] += u, u2[0] += u*u;
            c1[0] += c, c2[0] += c*c;
        }
        free(w_int), free(w_dbl);
        vslDeleteStream(&s);
    }
    *m1 /= m, *m2 /= m;
    *ma1 /= m, *ma2 /= m;
    *u1 /= m, *u2 /= m;
    *c1 /= m, *c2 /= m;
    return ;
}
