#include "samp.h"

void iswap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
    return ;
}

void pop_and_push_2d(int size, int x, int y, int* q_c, int* q_coor, int ctr_c[10], int* c_stack_, int inc)
{
    int n = size;
    int(* q_c_)[n] = (void*)q_c, (* q_coor_)[n] = (void*)q_coor;
    int(* c_stack)[n*n] = (void*)c_stack_;
    int c = q_c_[x][y], coor = q_coor_[x][y];
    iswap(c_stack[c]+coor, c_stack[c]+ctr_c[c]-1);
    q_coor[c_stack[c][coor]] = coor;
    ctr_c[c]--;
    c += inc;
    q_c_[x][y] = c;
    q_coor_[x][y] = ctr_c[c];
    c_stack[c][ctr_c[c]++] = x*n + y;
    return ;
}

void pop_and_push_3d(int size, int x, int y, int z, int* q_c, int* q_coor, int ctr_c[14], int* c_stack_, int inc)
{
    int n = size;
    int(* q_c_)[n][n] = (void*)q_c, (* q_coor_)[n][n] = (void*)q_coor;
    int(* c_stack)[n*n*n] = (void*)c_stack_;
    int c = q_c_[x][y][z], coor = q_coor_[x][y][z];
    iswap(c_stack[c]+coor, c_stack[c]+ctr_c[c]-1);
    q_coor[c_stack[c][coor]] = coor;
    ctr_c[c]--;
    c += inc;
    q_c_[x][y][z] = c;
    q_coor_[x][y][z] = ctr_c[c];
    c_stack[c][ctr_c[c]++] = x*n*n + y*n + z;
    return ;
}
