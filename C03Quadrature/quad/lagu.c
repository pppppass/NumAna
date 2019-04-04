#include "quad.h"

double calc_lagu(int order, double x)
{
    int k = order;
    if (k <= 0)
        return 1.0;
    double f_pre = 1.0, f = 1.0 - x;
    for (int i = 1; i < k; i++)
    {
        double f_next = (1.0 + 2.0 * (double)i - x) * f - (double)i * (double)i * f_pre;
        f_pre = f;
        f = f_next;
        // trick
        if (!isfinite(f))
            return INFINITY;
    }
    return f;
}

// Normalized version $ L_k (x) / k ! $
double calc_lagu_norm(int order, double x)
{
    int k = order;
    if (k <= 0)
        return 1.0;
    double f_pre = 1.0, f = 1.0 - x;
    for (int i = 1; i < k; i++)
    {
        double f_next = (1.0 + 2.0 * (double)i - x) / (1.0 + (double)i) * f - (double)i / (1.0 + (double)i) * f_pre;
        f_pre = f;
        f = f_next;
        // trick
        if (!isfinite(f))
            return INFINITY;
    }   
    return f;
}

//  zeros <OUT>:
//      [0:k]
//          [0:k]: d <TEMP>
//  work <TEMP>:
//      [0:k]: e
void calc_lagu_zero(int order, double* zeros, double* work)
{
    int k = order;
    double* d = zeros, * e = work;
    for (int i = 0; i < k; i++)
    {
        d[i] = 1.0 + 2.0 * (double)i;
        e[i] = 1.0 + (double)i;
    }
    // solve in-place: zeros[i] = d[i]
    LAPACKE_dstev(LAPACK_ROW_MAJOR, 'N', k, d, e, NULL, k);
    return ;
}

//  zeros:
//      [0:k]
//  weight <OUT>:
//      [0:k]
void calc_lagu_weight(int order, const double* zeros, double* weight)
{
    int k = order;
    const double* z = zeros;
    double* w = weight;
    for (int i = 0; i < k; i++)
    {
        double f = calc_lagu_norm(k+1, z[i]);
        w[i] = 1.0 / (1.0 + (double)k) / f;
        w[i] = w[i]*w[i] * z[i];
    }
    return ;
}

//  zeros <OUT>, weight <OUT>:
//      [0:k]
void calc_lagu_para(int order, double* zeros, double* weight)
{
    calc_lagu_zero(order, zeros, weight);
    calc_lagu_weight(order, zeros, weight);
    return ;
}

// //  x_node, y_node:
// //      [0:n+1]
// double quad_lagu(int num_node, const double* x_node, const double* y_node)
// {
//     int n = num_node;
//     double r = 0.0;
// #ifdef DEBUG
//     {
//         FILE* fout = fopen("Debug.log", "w+");
//         fprintf(fout, "Weights in Laguarre-Gauss:\n");
// #endif
//         for (int i = 0; i < n; i++)
//         {
//             // double w = exp(lgamma(1.0 + (double)n)) / calc_lagu(n+1, x_node[i]);
//             double f = calc_lagu_norm(n+1, x_node[i]);
//             // dirty trick for large n
//             double w = isnormal(f) ? (1.0 / (1.0 + (double)n) / f) : -1.0;
//             w = w*w * x_node[i];
//             r += w * y_node[i];
// #ifdef DEBUG
//             fprintf(fout, "%le %d\n", w, isnan(w));
// #endif
//         }
// #ifdef DEBUG
//         fprintf(fout, "\n");
//         fclose(fout);
//     }
// #endif
//     return r;
// }
