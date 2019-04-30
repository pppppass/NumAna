#include "fft.h"

//  u <OUT>:
//      [0:2^k]
void trans_fft(int size, double* vec)
{
    int k = size;
    double* u = vec;
    trans_trans(k, vec);
    for (int p = 0; p < k; p++)
    {
        for (int i = 0; i < 1<<(k-1-p); i++)
        {
            int x = i<<(p+1), y = (i<<(p+1))|(1<<p);
            double a = u[x], b = u[y];
            u[x] = a + b;
            u[y] = a - b;
            
            if (p >= 1)
            {
                int x = (i<<(p+1))|(1<<(p-1)), y = (i<<(p+1))|(3<<(p-1));
                double a = u[x], b = u[y];
                u[x] = a;
                u[y] = -b;
            }

            if (p >= 2)
            {
                int x = (i<<(p+1))|(1<<(p-2)), y = (i<<(p+1))|(1<<(p-2))|(1<<(p-1)), z = (i<<(p+1))|(1<<(p-2))|(1<<p), w = (i<<(p+1))|(1<<(p-2))|(3<<(p-1));
                double a = u[x], b = u[y], c = u[z], d = u[w];
                double e = M_SQRT1_2 * (c + d), f = M_SQRT1_2 * (-c + d);
                u[x] = a + e;
                u[y] = a - e;
                u[z] = b + f;
                u[w] = -b + f;
            }

            if (p >= 3)
            {
// #ifdef DEBUG
//                 FILE* file = fopen("Debug.log", "w");
// #endif
                for (int j = 1; j < 1<<(p-2); j++)
                {
                    int x1 = (i<<(p+1))|j, y1 = (i<<(p+1))|j|(1<<(p-1)), z1 = (i<<(p+1))|j|(1<<p), w1 = (i<<(p+1))|j|(3<<(p-1)), x2 = (i<<(p+1))|((1<<(p-1))-j), y2 = (i<<(p+1))|((1<<(p-1))-j)|(1<<(p-1)), z2 = (i<<(p+1))|((1<<(p-1))-j)|(1<<p), w2 = (i<<(p+1))|((1<<(p-1))-j)|(3<<(p-1));
                    double theta1 = 2.0 * M_PI * (double)j / (double)(1<<(p+1)), theta2 = 2.0 * M_PI * (double)((1<<(p-1))-j) / (double)(1<<(p+1));
                    double alpha1 = cos(theta1), beta1 = -sin(theta1), alpha2 = cos(theta2), beta2 = -sin(theta2);
                    double a1 = u[x1], b1 = u[y1], c1 = u[z1], d1 = u[w1], a2 = u[x2], b2 = u[y2], c2 = u[z2], d2 = u[w2];
                    double e1 = alpha1 * c1 - beta1 * d1, f1 = beta1 * c1 + alpha1 * d1, e2 = alpha2 * c2 - beta2 * d2, f2 = beta2 * c2 + alpha2 * d2;
                    u[x1] = a1 + e1;
                    u[y2] = a1 - e1;
                    u[z1] = b1 + f1;
                    u[w2] = -b1 + f1;
                    u[x2] = a2 + e2;
                    u[y1] = a2 - e2;
                    u[z2] = b2 + f2;
                    u[w1] = -b2 + f2;
    // #ifdef DEBUG
    //                     fprintf(file, "j = %d:\n", j);
    //                     fprintf(file, "%d, %d, %d, %d -> %d, %d, %d, %d", x, y, w, r, x, z, w, s);
    //                     fprintf(file, "x, y, z, w, r, s  = %d, %d, %d, %d, %d, %d\n", x, y, z, w, r, s);
    //                     fprintf(file, "a, b, c, d, e, f = %lf, %lf, %lf, %lf, %lf, %lf\n", a, b, c, d, e, f);
    //                     fprintf(file, "alpha, beta = %lf, %lf\n", alpha, beta);
    //                     fprintf(file, "u[x], u[y], u[z], u[w], u[r], u[s] = %lf, %lf, %lf, %lf, %lf, %lf\n", u[x], u[y], u[z], u[w], u[r], u[s]);
    //                     fprintf(file, "\n");
    // #endif
                }
// #ifdef DEBUG
//                 fclose(file);
// #endif
            }
        }
    }
    return ;
}

