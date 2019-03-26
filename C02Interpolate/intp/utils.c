#include "intp.h"

int i_clip(const int val, const int low, const int high)
{
    int x = val, l = low, h = high;
    if (x < l)
        return l;
    if (x >= h)
        return h-1;
    else
        return x;
}

int index_intv_unif(const int num_node, const double x_low, const double x_high, const double x)
{
    int n = num_node;
    int i = (int)((x - x_low) / (x_high - x_low) * (double)n);
    i = i_clip(i, 0, n);
    return i;
}

double coor_node_unif(const int num_node, const double x_low, const double x_high, const int index)
{
    int n = num_node;
    double x = x_low + (x_high - x_low) * (double)index / (double)n;
    return x;
}

// No C++-style lower_bound in C, so write one
int d_lower_bound(const int len, const double* arr, const double val)
{
    int n = len;
    const double* a = arr, x = val;
    int l = 0, h = n;
    while (h - l > 1)
    {
        int p = (l + h) / 2;
        if (x < a[p])
            h = p;
        else
            l = p;
    }
    return l;
}

double cvrt_ref(const double x_low, const double x_high, const double x)
{
    double x_ref = (x - x_low) / (x_high - x_low);
    return x_ref;
}

int index_intv_bis(const int num_node, const double* x_node, const double x)
{
    int n = num_node;
    int i = d_lower_bound(n, x_node, x);
    return i;
}
