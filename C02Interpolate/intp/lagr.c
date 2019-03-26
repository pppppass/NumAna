#include "intp.h"

// Brute force expansion to Lagrange expansion
void intp_lagr(int num_node, const double* x_node, const double* y_node, int num_req, const double* x_req, double* y_req)
{
    int n = num_node, m = num_req;
    for (int j = 0; j < m; j++)
        y_req[j] = 0.0;
    for (int i = 0; i <= n; i++)
        for (int j = 0; j < m; j++)
        {
            double c = y_node[i];
            for (int k = 0; k <= n; k++)
                if (k == i)
                    continue;
                else
                    c *= (x_req[j] - x_node[k]) / (x_node[i] - x_node[k]);
            y_req[j] += c;
        }
    return ;
}
