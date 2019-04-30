#include "fft.h"

unsigned int calc_rev_bit_32(unsigned int num)
{
    unsigned int v = num;
    // from https://graphics.stanford.edu/~seander/bithacks.html
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    v = ( v >> 16             ) | ( v               << 16);
    return v;
}

unsigned int calc_rev_bit(int len, unsigned int num)
{
    int k = len;
    unsigned int v = num;
    return calc_rev_bit_32(v) >> (32 - k);
}

//  vec <OUT>:
//      [0:2^k] 
void trans_trans(int size, double* vec)
{
    int k = size;
    double* u = vec;
    int n = 1<<k;
    for (int i = 0, j = 0; i < n; i++)
        if ((j = calc_rev_bit(k, i)) > i)
        {
            double t = u[j];
            u[j] = u[i];
            u[i] = t;
        }
    return ;
}
