#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import opt
from matplotlib import pyplot
import mpl_toolkits.mplot3d


# In[2]:


def gen_res_2d(d, extra=[]):
    res = [(i, j) for i in range(d+1) for j in range(d+1) if i + j <= d] + extra
    l = len(res)
    i, j = (numpy.array([e[t] for e in res], dtype=numpy.int32) for t in (0, 1))
#     print("L = {}".format(l))
#     print(res)
    return l, i, j


# In[3]:


def gen_init_2d(n, seed):
    numpy.random.seed(seed)
    sigma, alpha, beta = 0.01, 1.0/3.0, 1.0/n
    (x, y), w = (numpy.random.randn(n) * sigma + alpha for _ in range(2)), numpy.random.rand(n) * beta
    return x, y, w


# In[4]:


def plot_fig_2d(n, x, y, w, filename):
    pyplot.figure(figsize=(4.0, 4.0))
    pyplot.subplot(1, 1, 1, aspect="equal")
    for t in range(n):
        pyplot.scatter(x[t], y[t], color="C{}".format(t), s=10.0)
        pyplot.text(x[t], y[t], "{:.5f}".format(w[t]))
    tri = pyplot.Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    tri.set_edgecolor((0.0, 0.0, 0.0, 1.0))
    tri.set_linewidth(0.3)
    tri.set_facecolor((0.1, 0.1, 0.1, 0.1))
    pyplot.gca().add_patch(tri)
    pyplot.xlim(-0.2, 1.2)
    pyplot.ylim(-0.2, 1.2)
    pyplot.grid()
    if filename is not None:
        pyplot.savefig(filename)
    pyplot.show()
    pyplot.close()


# In[5]:


n = 1
d = 1
c = numpy.array([0], dtype=numpy.int32)
l, i, j = gen_res_2d(d)
x, y, w = gen_init_2d(n, 1)
opt.opt_gauss_2d_newt(n, l, i, j, numpy.max(c)+1, c, x, y, w, 1.0, 30)
f = opt.calc_val_sos_2d_dist(n, l, i, j, x, y, w)
print(f)
plot_fig_2d(n, x, y, w, "Figure01.pgf")


# In[6]:


n = 2
d = 2
c = numpy.array([0, 1], dtype=numpy.int32)
l, i, j = gen_res_2d(d)
x, y, w = gen_init_2d(n, 1)
opt.opt_gauss_sos_2d_grad_nest(n, l, i, j, numpy.max(c)+1, c, x, y, w, 0.01, 1000000)
f = opt.calc_val_sos_2d_dist(n, l, i, j, x, y, w)
print(f)
with open("Text1.txt", "w") as file:
    file.write("{:.5e}".format(f))


# In[7]:


seeds = [1, 2, 4, 5, 8, 12]
for t, s in enumerate(seeds):
    n = 3
    d = 2
    c = numpy.array([0, 0, 0], dtype=numpy.int32)
    l, i, j = gen_res_2d(d)
    x, y, w = gen_init_2d(n, s)
    opt.opt_gauss_sos_2d_grad_nest(n, l, i, j, numpy.max(c)+1, c, x, y, w, 0.01, 100000)
    f = opt.calc_val_sos_2d_dist(n, l, i, j, x, y, w)
    print(f)
    plot_fig_2d(n, x, y, w, "Figure{:02}.pgf".format(t+2))


# In[8]:


seeds = [1, 7]
for t, s in enumerate(seeds):
    n = 3
    d = 2
    c = numpy.array([0, 0, 0], dtype=numpy.int32)
    l, i, j = gen_res_2d(d, [(0, 3)])
    x, y, w = gen_init_2d(n, s)
    opt.opt_gauss_sos_2d_grad_nest(n, l, i, j, numpy.max(c)+1, c, x, y, w, 0.01, 10000)
    opt.opt_gauss_2d_newt(n, l, i, j, numpy.max(c)+1, c, x, y, w, 1.0, 30)
    f = opt.calc_val_sos_2d_dist(n, l, i, j, x, y, w)
    print(f)
    plot_fig_2d(n, x, y, w, "Figure{:02}.pgf".format(t+8))


# In[22]:


seeds = [1, 5]
for t, s in enumerate(seeds):
    n = 3
    d = 2
    c = numpy.array([0, 0, 0], dtype=numpy.int32)
    l, i, j = gen_res_2d(d, [(1, 2)])
    x, y, w = gen_init_2d(n, s)
    opt.opt_gauss_sos_2d_grad_nest(n, l, i, j, numpy.max(c)+1, c, x, y, w, 0.01, 10000)
    opt.opt_gauss_2d_newt(n, l, i, j, numpy.max(c)+1, c, x, y, w, 1.0, 30)
    f = opt.calc_val_sos_2d_dist(n, l, i, j, x, y, w)
    print(f)
    plot_fig_2d(n, x, y, w, "Figure{:02}.pgf".format(t+10))


# In[10]:


n = 3
d = 3
c = numpy.array([0, 1, 2], dtype=numpy.int32)
l, i, j = gen_res_2d(d)
x, y, w = gen_init_2d(n, 1)
opt.opt_gauss_sos_2d_grad_nest(n, l, i, j, numpy.max(c)+1, c, x, y, w, 0.01, 1000000)
f = opt.calc_val_sos_2d_dist(n, l, i, j, x, y, w)
print(f)
with open("Text2.txt", "w") as file:
    file.write("{:.5e}".format(f))


# In[11]:


seeds = [1, 3, 16]
for t, s in enumerate(seeds):
    n = 4
    d = 3
    c = numpy.array([0, 0, 1, 1], dtype=numpy.int32)
    l, i, j = gen_res_2d(d, [(1, 2)])
    x, y, w = gen_init_2d(n, s)
    opt.opt_gauss_sos_2d_grad_nest(n, l, i, j, numpy.max(c)+1, c, x, y, w, 0.01, 10000)
    opt.opt_gauss_2d_newt(n, l, i, j, numpy.max(c)+1, c, x, y, w, 1.0, 30)
    f = opt.calc_val_sos_2d_dist(n, l, i, j, x, y, w)
    print(f)
    plot_fig_2d(n, x, y, w, "Figure{:02}.pgf".format(t+12))


# In[12]:


def gen_res_3d(d, extra=[]):
    res = [(i, j, k) for i in range(d+1) for j in range(d+1) for k in range(d+1) if i + j + k <= d] + extra
    l = len(res)
    i, j, k = (numpy.array([e[t] for e in res], dtype=numpy.int32) for t in (0, 1, 2))
    return l, i, j, k


# In[13]:


def gen_init_3d(n, seed):
    numpy.random.seed(seed)
    sigma, alpha, beta = 0.01, 1.0/4.0, 1.0/3.0/n
    (x, y, z), w = (numpy.random.randn(n) * sigma + alpha for _ in range(3)), numpy.random.rand(n) * beta
    return x, y, z, w


# In[14]:


def draw_ref_tetra(): 
    tri = mpl_toolkits.mplot3d.art3d.Poly3DCollection([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    ])
    tri.set_edgecolor((0.0, 0.0, 0.0, 1.0))
    tri.set_linewidth(0.3)
    tri.set_facecolor((0.1, 0.1, 0.1, 0.1))
    return tri


# In[15]:


def plot_fig_3d(n, x, y, z, w, filename):
    pyplot.figure(figsize=(8.0, 4.0))
    pyplot.subplot(1, 2, 1, projection="3d", aspect="equal")
    tri = draw_ref_tetra()
    pyplot.gca().add_collection(tri)
    for t in range(n):
        pyplot.gca().scatter(x[t], y[t], z[t], color="C{}".format(t))
        pyplot.gca().text(x[t], y[t], z[t], "{:.5f}".format(w[t]))
    pyplot.gca().view_init(azim=60.0)
    pyplot.subplot(1, 2, 2, projection="3d", aspect="equal")
    tri = draw_ref_tetra()
    tri.set_edgecolor((0.0, 0.0, 0.0, 1.0))
    tri.set_linewidth(0.3)
    tri.set_facecolor((0.1, 0.1, 0.1, 0.1))
    pyplot.gca().add_collection(tri)
    for t in range(n):
        pyplot.gca().scatter(x[t], y[t], z[t], color="C{}".format(t))
        pyplot.gca().text(x[t], y[t], z[t], "{:.5f}".format(w[t]))
    pyplot.gca().view_init(azim=-60.0)
    if filename is not None:
        pyplot.savefig(filename)
    pyplot.show()
    pyplot.close()


# In[24]:


n = 3
d = 2
c = numpy.array([0, 1, 2, 3], dtype=numpy.int32)
l, i, j, k = gen_res_3d(d)
x, y, z, w = gen_init_3d(n, 1)
opt.opt_gauss_sos_3d_grad_nest(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 0.01, 1000000)
f = opt.calc_val_sos_3d_dist(n, l, i, j, k, x, y, z, w)
print(f)
with open("Text3.txt", "w") as file:
    file.write("{:.5e}".format(f))


# In[17]:


seeds = [1, 2, 3, 4, 5, 6]
for t, s in enumerate(seeds):
    n = 4
    d = 2
    c = numpy.array([0, 0, 0, 0], dtype=numpy.int32)
    l, i, j, k = gen_res_3d(d)#, [(0, 0, 3), (0, 3, 0), (3, 0, 0)])
    x, y, z, w = gen_init_3d(n, s)
    opt.opt_gauss_sos_3d_grad_nest(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 0.01, 10000)
#     opt.opt_gauss_3d_newt(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 1.0, 30)
    f = opt.calc_val_sos_3d_dist(n, l, i, j, k, x, y, z, w)
    print(f)
    plot_fig_3d(n, x, y, z, w, "Figure{:02}.pgf".format(t+15))


# In[26]:


seeds = [1, 2]
for t, s in enumerate(seeds):
    n = 4
    d = 2
    c = numpy.array([0, 0, 0, 0], dtype=numpy.int32)
    l, i, j, k = gen_res_3d(d, [(0, 0, 3), (0, 3, 0), (3, 0, 0)])
    x, y, z, w = gen_init_3d(n, s)
    opt.opt_gauss_sos_3d_grad_nest(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 0.01, 10000)
    opt.opt_gauss_3d_newt(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 1.0, 30)
    f = opt.calc_val_sos_3d_dist(n, l, i, j, k, x, y, z, w)
    print(f)
    plot_fig_3d(n, x, y, z, w, "Figure{:02}.pgf".format(t+21))


# In[19]:


n = 5
d = 3
c = numpy.array([0, 1, 2, 3, 4], dtype=numpy.int32)
l, i, j, k = gen_res_3d(d)
x, y, z, w = gen_init_3d(n, 1)
opt.opt_gauss_sos_3d_grad_nest(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 0.01, 1000000)
f = opt.calc_val_sos_3d_dist(n, l, i, j, k, x, y, z, w)
print(f)
with open("Text4.txt", "w") as file:
    file.write("{:.5e}".format(f))


# In[27]:


seeds = [14, 40]
for t, s in enumerate(seeds):
    n = 6
    d = 3
    c = numpy.array([0, 0, 0, 1, 1, 1], dtype=numpy.int32)
    l, i, j, k = gen_res_3d(d)
    x, y, z, w = gen_init_3d(n, s)
    opt.opt_gauss_sos_3d_grad_nest(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 0.01, 10000)
    opt.opt_gauss_3d_newt(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 1.0, 30)
    f = opt.calc_val_sos_3d_dist(n, l, i, j, k, x, y, z, w)
    print(f)
    plot_fig_3d(n, x, y, z, w, "Figure{:02}.pgf".format(t+23))


# In[23]:


n = 1
d = 1
c = numpy.array([0], dtype=numpy.int32)
l, i, j, k = gen_res_3d(d)
x, y, z, w = gen_init_3d(n, s)
opt.opt_gauss_3d_newt(n, l, i, j, k, numpy.max(c)+1, c, x, y, z, w, 1.0, 30)
f = opt.calc_val_sos_3d_dist(n, l, i, j, k, x, y, z, w)
print(f)
plot_fig_3d(n, x, y, z, w, "Figure14.pgf")


# In[ ]:




