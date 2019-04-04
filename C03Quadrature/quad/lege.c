#include "quad.h"

double calc_lege(int order, double x)
{
    int k = order;
    if (k <= 0)
        return 1.0;
    double f_pre = 1.0, f = x;
    for (int i = 1; i < k; i++)
    {
        double f_next = (1.0 + 2.0 * (double)i) / (1.0 + (double)i) * x * f - (double)i / (1.0 + (double)i) * f_pre;
        f_pre = f;
        f = f_next;
    }
    return f;
}

//  zeros <OUT>:
//      [0:k]
//          [0:k]: d <TEMP>
//  work <TEMP>:
//      [k:k]: e
void calc_lege_zero(int order, double* zeros, double* work)
{
    int k = order;
    double* z = zeros;
    double* d = zeros, * e = work;
    for (int i = 0; i < k; i++)
    {
        d[i] = 0.0;
        e[i] = (1.0 + (double)i) / sqrt((2.0 * (double)i + 1.0) * (2.0 * (double)i + 3.0));
    }
    // solve in-place: zeros[i] = d[i]
    int r = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'N', k, d, e, NULL, k);
#ifdef DEBUG
    {
        FILE* fout = fopen("Debug.log", "w+");
        fprintf(fout, "Return value of dstev:\n");
        fprintf(fout, "%d\n", r);
        fprintf(fout, "\n");
        fclose(fout);
    }
#endif
    // for (int i = 0; i < k; i++)
    //     z[i] = (x_low + x_high) / 2.0 + z[i] * (x_high - x_low) / 2.0;
    return ;
}

//  zeros:
//      [0:k]
//  weight <OUT>:
//      [0:k]
void calc_lege_weight(int order, const double* zeros, double* weight)
{
    int k = order;
    const double* z = zeros;
    double* w = weight;
    for (int i = 0; i < k; i++)
    {
        double f = calc_lege(k-1, z[i]);
        w[i] = 1.0 / (double)k / f;
        w[i] = w[i]*w[i] * (1.0 - z[i]*z[i]);
    }
    return ;
}

//  zeros <OUT>, weight <OUT>:
//      [0:k]
void calc_lege_para(int order, double x_low, double x_high, double* zeros, double* weight)
{
    int k = order;
    double* z = zeros, * w = weight;
    calc_lege_zero(order, zeros, weight);
    calc_lege_weight(order, zeros, weight);
    for (int i = 0; i < k; i++)
    {
        z[i] = (x_low + x_high) / 2.0 + z[i] * (x_high - x_low) / 2.0;
        w[i] *= (x_high - x_low);
    }
    return ;
}

// //  x_node, y_node:
// //      [0:n+1]
// double quad_lege(int num_node, double x_low, double x_high, const double* x_node, const double* y_node)
// {
//     int n = num_node;
//     double r = 0.0;
//     for (int i = 0; i < n; i++)
//     {
//         double x = (x_node[i] - (x_low + x_high) / 2.0) / (x_low - x_high) * 2.0;
//         double w = 1.0 / (double)n / calc_lege(n-1, x);
//         w = (1.0 - x*x) * w*w;
//         r += w * y_node[i];
//     }
//     r *= (x_high - x_low);
//     return r;
// }
