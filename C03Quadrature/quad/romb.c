#include "quad.h"
#ifdef DEBUG
#include <stdio.h>
#endif

//  y_node:
//      [0:2^k*n+1]
//  work <TEMP>:
//      [0:k+1]
double quad_romb_unif(int num_node, int num_order, double x_low, double x_high, const double* y_node, double* work)
{
    int n = num_node, k = num_order;
    double* c = work;
    // k is always small and we calculate it whenever needed
    c[0] = 1.0;
    {
        // accumulator for $ 2^{ 2 k - 1 } $
        double acc = 2.0;
        for (int i = 1; i <= k; i++)
        {
            c[i] = 0.0;
            for (int j = i; j > 0; j--)
                c[j] = (acc * c[j-1] - c[j]) / (2.0 * acc - 1.0);
            c[0] *= (acc - 1.0) / (2.0 * acc - 1.0);
            acc *= 4.0;
        }
    }
#ifdef DEBUG
    {
        FILE* fout = fopen("Debug.log", "w+");
        fprintf(fout, "Data in c[i]:\n");
        for (int j = 0; j <= k; j++)
            fprintf(fout, "%lf\n", c[j]);
        fprintf(fout, "\n");
        fclose(fout);
    }
#endif
    double h = (x_high - x_low) / n;
    double r = 0.0;
    {
        double acc = 0.0;
        acc += (y_node[0] + y_node[n<<k]) / 2.0;
        for (int j = 1; j < n; j++)
            acc += y_node[j<<k];
        r += c[0] * acc;
    }
    for (int i = 1; i <= k; i++)
    {
        double acc = 0.0;
        for (int j = 0; j < n<<(i-1); j++)
            acc += y_node[j<<(k-i+1) | 1<<(k-i)];
        r += c[i] * acc;
    }
    r *= h;
    return r;
}
