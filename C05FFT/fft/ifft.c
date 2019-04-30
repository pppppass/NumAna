#include "fft.h"

//  u <OUT>:
//      [0:2^k]
void trans_ifft(int size, double* vec)
{
    int k = size;
    double* u = vec;
    for (int p = k-1; p >= 0; p--)
    {
        for (int i = 0; i < 1<<(k-1-p); i++)
        {
            int x = i<<(p+1), y = (i<<(p+1))|(1<<p);
            // a_ = a + b, b_ = a - b
            double a_ = u[x], b_ = u[y];
            u[x] = (a_ + b_) / 2.0;
            u[y] = (a_ - b_) / 2.0;
            
            if (p >= 1)
            {
                int x = (i<<(p+1))|(1<<(p-1)), y = (i<<(p+1))|(3<<(p-1));
                // a_ = a, b_ = -b
                double a_ = u[x], b_ = u[y];
                u[x] = a_;
                u[y] = -b_;
            }

            if (p >= 2)
            {
                int x = (i<<(p+1))|(1<<(p-2)), y = (i<<(p+1))|(1<<(p-2))|(1<<(p-1)), z = (i<<(p+1))|(1<<(p-2))|(1<<p), w = (i<<(p+1))|(1<<(p-2))|(3<<(p-1));
                // a_ = a + e, e_ = a - e, b_ = b + f, f_ = -b + f
                double a_ = u[x], e_ = u[y], b_ = u[z], f_ = u[w];
                // e = sqrt(1/2) * (c + d), f = sqrt(1/2) * (-c + d)
                double e = (a_ - e_) / 2.0, f = (b_ + f_) / 2.0;
                u[x] = (a_ + e_) / 2.0;
                u[y] = (b_ - f_) / 2.0;
                u[z] = M_SQRT1_2 * (e - f);
                u[w] = M_SQRT1_2 * (e + f);
            }

            if (p >= 3)
            {
                for (int j = 1; j < 1<<(p-2); j++)
                {
                    int x1 = (i<<(p+1))|j, y1 = (i<<(p+1))|j|(1<<(p-1)), z1 = (i<<(p+1))|j|(1<<p), w1 = (i<<(p+1))|j|(3<<(p-1)), x2 = (i<<(p+1))|((1<<(p-1))-j), y2 = (i<<(p+1))|((1<<(p-1))-j)|(1<<(p-1)), z2 = (i<<(p+1))|((1<<(p-1))-j)|(1<<p), w2 = (i<<(p+1))|((1<<(p-1))-j)|(3<<(p-1));
                    double theta1 = 2.0 * M_PI * (double)j / (double)(1<<(p+1)), theta2 = 2.0 * M_PI * (double)((1<<(p-1))-j) / (double)(1<<(p+1));
                    double alpha1 = cos(theta1), beta1 = sin(theta1), alpha2 = cos(theta2), beta2 = sin(theta2);
                    // a_ = a + e, e_ = a - e, b_ = b + f, f_ = -b + f
                    double a_1 = u[x1], e_1 = u[y2], b_1 = u[z1], f_1 = u[w2], a_2 = u[x2], e_2 = u[y1], b_2 = u[z2], f_2 = u[w1];
                    // e = alpha * c + beta * d, f = -beta * c + alpha * d
                    // note: sign of beta altered
                    double e1 = (a_1 - e_1) / 2.0, f1 = (b_1 + f_1) / 2.0, e2 = (a_2 - e_2) / 2.0, f2 = (b_2 + f_2) / 2.0;
                    u[x1] = (a_1 + e_1) / 2.0;
                    u[y1] = (b_1 - f_1) / 2.0;
                    u[z1] = alpha1 * e1 - beta1 * f1;
                    u[w1] = beta1 * e1 + alpha1 * f1;
                    u[x2] = (a_2 + e_2) / 2.0;
                    u[y2] = (b_2 - f_2) / 2.0;
                    u[z2] = alpha2 * e2 - beta2 * f2;
                    u[w2] = beta2 * e2 + alpha2 * f2;
                }
            }
        }
    }
    trans_trans(k, vec);
    return ;
}
