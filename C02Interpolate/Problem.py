#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
import intp


# In[2]:


f = lambda x: 1.0 / (1.0 + x**2)
d_f = lambda x: -2.0 * x / (1.0 + x**2)**2
d_d_f = lambda x: (6.0 * x**2 - 2.0) / (1.0 + x**2)**3


# In[3]:


n = 5
m = 12000
h = 12.0 / m


# In[4]:


x_node = numpy.linspace(-5.0, 5.0, n+1)
y_node = f(x_node)
d_y_node = d_f(x_node)


# In[5]:


x_req = numpy.linspace(-6.0, 6.0, m+1)
y_req = f(x_req)


# In[6]:


y_intp_lagr = intp.intp_lagr(n, x_node, y_node, m+1, x_req, numpy.zeros(m+1))
y_intp_lin = intp.intp_lin_unif(n, -5.0, 5.0, y_node, m+1, x_req, numpy.zeros(m+1))
y_intp_cub = intp.intp_cub(n, x_node, y_node, d_y_node, m+1, x_req, numpy.zeros(m+1))
d_y_spl_cub_nat = intp.calc_spl_cub_d_y_nat(n, x_node, y_node, numpy.zeros(n+1))
y_intp_spl_cub_nat = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_nat, m+1, x_req, numpy.zeros(m+1))
d_y_spl_cub_coe = intp.calc_spl_cub_d_y_coe(n, x_node, y_node, *d_y_node[[0, -1]], numpy.zeros(n+1))
y_intp_spl_cub_coe = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_coe, m+1, x_req, numpy.zeros(m+1))


# In[7]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.scatter(x_node, y_node, s=5.0)
pyplot.plot(x_req, y_req, label="Runge", linewidth=1.0)
pyplot.plot(x_req, y_intp_lagr, label="Lagrange", linewidth=1.0)
pyplot.plot(x_req, y_intp_lin, label="Linear", linewidth=1.0)
pyplot.plot(x_req, y_intp_cub, label="Cubic", linewidth=1.0)
pyplot.plot(x_req, y_intp_spl_cub_nat, label="Spline (natural)", linewidth=1.0)
pyplot.plot(x_req, y_intp_spl_cub_coe, label="Spline (coercive)", linewidth=1.0)
pyplot.xlim(-6.0, 6.0)
pyplot.ylim(-0.5, 1.5)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure1.pgf")
pyplot.show()


# In[8]:


x_mid = (x_req[:-1] + x_req[1:]) / 2.0
d_y_mid = d_f(x_mid)
d_y_intp_cub = (y_intp_cub[1:] - y_intp_cub[:-1]) / h
d_y_intp_spl_cub_nat = (y_intp_spl_cub_nat[1:] - y_intp_spl_cub_nat[:-1]) / h
d_y_intp_spl_cub_coe = (y_intp_spl_cub_coe[1:] - y_intp_spl_cub_coe[:-1]) / h


# In[9]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.plot(x_mid, d_y_mid, label="Runge", linewidth=1.0)
pyplot.plot(x_mid, d_y_intp_cub, label="Cubic", linewidth=1.0)
pyplot.plot(x_mid, d_y_intp_spl_cub_nat, label="Spline (natural)", linewidth=1.0)
pyplot.plot(x_mid, d_y_intp_spl_cub_coe, label="Spline (coercive)", linewidth=1.0)
pyplot.xlim(-6.0, 6.0)
pyplot.ylim(-0.75, 0.75)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure2.pgf")
pyplot.show()


# In[10]:


d_d_y_req = d_d_f(x_req)
d_d_y_intp_cub = (y_intp_cub[:-2] + y_intp_cub[2:] - 2.0 * y_intp_cub[1:-1]) / h**2
d_d_y_intp_spl_cub_nat = (y_intp_spl_cub_nat[:-2] + y_intp_spl_cub_nat[2:] - 2.0 * y_intp_spl_cub_nat[1:-1]) / h**2
d_d_y_intp_spl_cub_coe = (y_intp_spl_cub_coe[:-2] + y_intp_spl_cub_coe[2:] - 2.0 * y_intp_spl_cub_coe[1:-1]) / h**2


# In[11]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.plot(x_req[1:-1], d_d_y_req[1:-1], label="Runge", linewidth=1.0)
pyplot.plot(x_req[1:-1], d_d_y_intp_cub, label="Cubic", linewidth=1.0)
pyplot.plot(x_req[1:-1], d_d_y_intp_spl_cub_nat, label="Spline (natural)", linewidth=1.0)
pyplot.plot(x_req[1:-1], d_d_y_intp_spl_cub_coe, label="Spline (coercive)", linewidth=1.0)
pyplot.xlim(-6.0, 6.0)
pyplot.ylim(-2.2, 1.1)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure3.pgf")
pyplot.show()


# In[12]:


n = 10


# In[13]:


x_node = numpy.linspace(-5.0, 5.0, n+1)
y_node = f(x_node)
d_y_node = d_f(x_node)


# In[14]:


y_intp_lagr = intp.intp_lagr(n, x_node, y_node, m+1, x_req, numpy.zeros(m+1))
y_intp_lin = intp.intp_lin_unif(n, -5.0, 5.0, y_node, m+1, x_req, numpy.zeros(m+1))
y_intp_cub = intp.intp_cub(n, x_node, y_node, d_y_node, m+1, x_req, numpy.zeros(m+1))
d_y_spl_cub_nat = intp.calc_spl_cub_d_y_nat(n, x_node, y_node, numpy.zeros(n+1))
y_intp_spl_cub_nat = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_nat, m+1, x_req, numpy.zeros(m+1))
d_y_spl_cub_coe = intp.calc_spl_cub_d_y_coe(n, x_node, y_node, *d_y_node[[0, -1]], numpy.zeros(n+1))
y_intp_spl_cub_coe = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_coe, m+1, x_req, numpy.zeros(m+1))


# In[15]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.scatter(x_node, y_node, s=5.0)
pyplot.plot(x_req, y_req, label="Runge", linewidth=1.0)
pyplot.plot(x_req, y_intp_lagr, label="Lagrange", linewidth=1.0)
pyplot.plot(x_req, y_intp_lin, label="Linear", linewidth=1.0)
pyplot.plot(x_req, y_intp_cub, label="Cubic", linewidth=1.0)
pyplot.plot(x_req, y_intp_spl_cub_nat, label="Spline (natural)", linewidth=1.0)
pyplot.plot(x_req, y_intp_spl_cub_coe, label="Spline (coercive)", linewidth=1.0)
pyplot.xlim(-6.0, 6.0)
pyplot.ylim(-0.5, 1.5)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure4.pgf")
pyplot.show()


# In[16]:


n = 20


# In[17]:


x_node = numpy.linspace(-5.0, 5.0, n+1)
y_node = f(x_node)
d_y_node = d_f(x_node)


# In[18]:


y_intp_lagr = intp.intp_lagr(n, x_node, y_node, m+1, x_req, numpy.zeros(m+1))
y_intp_lin = intp.intp_lin_unif(n, -5.0, 5.0, y_node, m+1, x_req, numpy.zeros(m+1))
y_intp_cub = intp.intp_cub(n, x_node, y_node, d_y_node, m+1, x_req, numpy.zeros(m+1))
d_y_spl_cub_nat = intp.calc_spl_cub_d_y_nat(n, x_node, y_node, numpy.zeros(n+1))
y_intp_spl_cub_nat = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_nat, m+1, x_req, numpy.zeros(m+1))
d_y_spl_cub_coe = intp.calc_spl_cub_d_y_coe(n, x_node, y_node, *d_y_node[[0, -1]], numpy.zeros(n+1))
y_intp_spl_cub_coe = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_coe, m+1, x_req, numpy.zeros(m+1))


# In[19]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.scatter(x_node, y_node, s=5.0)
pyplot.plot(x_req, y_req, label="Runge", linewidth=1.0)
pyplot.plot(x_req, y_intp_lagr, label="Lagrange", linewidth=1.0)
pyplot.plot(x_req, y_intp_lin, label="Linear", linewidth=1.0)
pyplot.plot(x_req, y_intp_cub, label="Cubic", linewidth=1.0)
pyplot.plot(x_req, y_intp_spl_cub_nat, label="Spline (natural)", linewidth=1.0)
pyplot.plot(x_req, y_intp_spl_cub_coe, label="Spline (coercive)", linewidth=1.0)
pyplot.xlim(-5.5, 5.5)
pyplot.ylim(-0.5, 1.5)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure5.pgf")
pyplot.show()


# In[20]:


n = 5


# In[21]:


theta = numpy.linspace(numpy.pi * (2*n+1) / (2*(n+1)), numpy.pi / (2*(n+1)), n+1)
x_node = 5.0 * numpy.cos(theta)
y_node = f(x_node)


# In[22]:


c = intp.calc_newt_arr(n, x_node, y_node, numpy.zeros(n+1))
y_intp_newt = intp.intp_newt(n, x_node, c, m+1, x_req, numpy.zeros(m+1))


# In[23]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.scatter(x_node, y_node, s=5.0)
pyplot.plot(x_req, y_req, label="Runge", linewidth=1.0)
pyplot.plot(x_req, y_intp_newt, label="Newton", linewidth=1.0)
pyplot.xlim(-6.0, 6.0)
pyplot.ylim(-0.5, 1.5)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure6.pgf")
pyplot.show()


# In[24]:


n = 10


# In[25]:


theta = numpy.linspace(numpy.pi * (2*n+1) / (2*(n+1)), numpy.pi / (2*(n+1)), n+1)
x_node = 5.0 * numpy.cos(theta)
y_node = f(x_node)


# In[26]:


c = intp.calc_newt_arr(n, x_node, y_node, numpy.zeros(n+1))
y_intp_newt = intp.intp_newt(n, x_node, c, m+1, x_req, numpy.zeros(m+1))


# In[27]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.scatter(x_node, y_node, s=5.0)
pyplot.plot(x_req, y_req, label="Runge", linewidth=1.0)
pyplot.plot(x_req, y_intp_newt, label="Newton", linewidth=1.0)
pyplot.xlim(-6.0, 6.0)
pyplot.ylim(-0.5, 1.5)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure7.pgf")
pyplot.show()


# In[28]:


n = 20


# In[29]:


theta = numpy.linspace(numpy.pi * (2*n+1) / (2*(n+1)), numpy.pi / (2*(n+1)), n+1)
x_node = 5.0 * numpy.cos(theta)
y_node = f(x_node)


# In[30]:


c = intp.calc_newt_arr(n, x_node, y_node, numpy.zeros(n+1))
y_intp_newt = intp.intp_newt(n, x_node, c, m+1, x_req, numpy.zeros(m+1))


# In[31]:


pyplot.figure(figsize=(8.0, 6.0))
pyplot.scatter(x_node, y_node, s=5.0)
pyplot.plot(x_req, y_req, label="Runge", linewidth=1.0)
pyplot.plot(x_req, y_intp_newt, label="Newton", linewidth=1.0)
pyplot.xlim(-6.0, 6.0)
pyplot.ylim(-0.5, 1.5)
pyplot.grid()
pyplot.legend()
pyplot.savefig("Figure8.pgf")
pyplot.show()


# In[51]:


f = lambda x: numpy.exp(x)


# In[52]:


n = 100
m = 1200


# In[53]:


x_node = numpy.linspace(0.0, 1.0, n+1)
y_node = f(x_node)
d_y_node = d_f(x_node)


# In[54]:


x_req = numpy.linspace(-1.0, 2.0, m+1)
y_req = f(x_req)


# In[55]:


ind = numpy.zeros(n+1, dtype=numpy.int)
ind[0], ind[1] = n, 0
for k in range(2, n+1):
    t = numpy.ones(n+1)
    for i in range(k):
        t *= (x_node - x_node[ind[i]])
    ind[k] = numpy.argmax(numpy.abs(t))


# In[56]:


y_intp_lagr = intp.intp_lagr(n, x_node, y_node, m+1, x_req, numpy.zeros(m+1))
c = intp.calc_newt_arr(n, x_node, y_node, numpy.zeros(n+1))
y_intp_newt = intp.intp_newt(n, x_node, c, m+1, x_req, numpy.zeros(m+1))
c_perm = intp.calc_newt_arr(n, x_node[ind], y_node[ind], numpy.zeros(n+1))
y_intp_newt_perm = intp.intp_newt(n, x_node[ind], c_perm, m+1, x_req, numpy.zeros(m+1))


# In[67]:


i_print = list(range(392, 409)) + list(range(420, 425))+ list(range(496, 501)) + list(range(596, 601))


# In[68]:


with open("Table1.tbl", "w") as file:
    for i in i_print:
        file.write("{:.4f} & {:.5e} & {:.5e} & {:.5e} \\\\\n\\hline\n".format(
            x_req[i],
            y_intp_lagr[i] - y_req[i],
            y_intp_newt[i] - y_req[i],
            y_intp_newt_perm[i] - y_req[i],
        ))


# In[69]:


i_print = list(range(600, 605)) + list(range(700, 705)) + list(range(776, 781)) + list(range(792, 809))


# In[70]:


with open("Table2.tbl", "w") as file:
    for i in i_print:
        file.write("{:.4f} & {:.5e} & {:.5e} & {:.5e} \\\\\n\\hline\n".format(
            x_req[i],
            y_intp_lagr[i] - y_req[i],
            y_intp_newt[i] - y_req[i],
            y_intp_newt_perm[i] - y_req[i],
        ))


# In[71]:


f = lambda x: 1.0 / (1.0 + x**2)
d_f = lambda x: -2.0 * x / (1.0 + x**2)**2


# In[43]:


n_list = list(range(2, 100))
m = 10000
h = 10.0 / m


# In[44]:


def sum_err(y_intp, y_req):
    err_inf = numpy.linalg.norm(y_intp - y_req, numpy.infty)
    err_2 = numpy.linalg.norm(y_intp - y_req, 2.0) * h**(1.0 / 2.0)
    err_1 = numpy.linalg.norm(y_intp - y_req, 1.0) * h
    return (err_1, err_2, err_inf)


# In[45]:


rt = [[], [], [], [], [], []]


# In[46]:


x_req = numpy.linspace(-5.0, 5.0, m+1)
y_req = f(x_req)


# In[47]:


for n in n_list:
    x_node = numpy.linspace(-5.0, 5.0, n+1)
    y_node = f(x_node)
    d_y_node = d_f(x_node)
    y_intp_lagr = intp.intp_lagr(n, x_node, y_node, m+1, x_req, numpy.zeros(m+1))
    rt[0].append(sum_err(y_intp_lagr, y_req))
    y_intp_lin = intp.intp_lin_unif(n, -5.0, 5.0, y_node, m+1, x_req, numpy.zeros(m+1))
    rt[1].append(sum_err(y_intp_lin, y_req))
    y_intp_cub = intp.intp_cub(n, x_node, y_node, d_y_node, m+1, x_req, numpy.zeros(m+1))
    rt[2].append(sum_err(y_intp_cub, y_req))
    d_y_spl_cub_nat = intp.calc_spl_cub_d_y_nat(n, x_node, y_node, numpy.zeros(n+1))
    y_intp_spl_cub_nat = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_nat, m+1, x_req, numpy.zeros(m+1))
    rt[3].append(sum_err(y_intp_spl_cub_nat, y_req))
    d_y_spl_cub_coe = intp.calc_spl_cub_d_y_coe(n, x_node, y_node, *d_y_node[[0, -1]], numpy.zeros(n+1))
    y_intp_spl_cub_coe = intp.intp_cub(n, x_node, y_node, d_y_spl_cub_coe, m+1, x_req, numpy.zeros(m+1))
    rt[4].append(sum_err(y_intp_spl_cub_coe, y_req))
    theta = numpy.linspace(numpy.pi * (2*n+1) / (2*(n+1)), numpy.pi / (2*(n+1)), n+1)
    x_node = 5.0 * numpy.cos(theta)
    y_node = f(x_node)
    y_intp_lagr = intp.intp_lagr(n, x_node, y_node, m+1, x_req, numpy.zeros(m+1))
    rt[5].append(sum_err(y_intp_lagr, y_req))


# In[48]:


def calc_log_line(start, end, intc, order):
    return [start, end], [intc, intc * (end / start)**order]


# In[49]:


titles = ["Lagrange (even)", "Linear", "Cubic", "Spline (natural)", "Spline (coercive)", "Lagrange (Chebyshev)"]


# In[50]:


pyplot.figure(figsize=(8.0, 12.0))
for i in range(6):
    pyplot.subplot(3, 2, i+1)
    pyplot.title(titles[i])
    pyplot.scatter(n_list, [e[0] for e in rt[i]], s=2.0)
    pyplot.plot(n_list, [e[0] for e in rt[i]], label="$L^1$")
    pyplot.scatter(n_list, [e[1] for e in rt[i]], s=2.0)
    pyplot.plot(n_list, [e[1] for e in rt[i]], label="$L^2$")
    pyplot.scatter(n_list, [e[2] for e in rt[i]], s=2.0)
    pyplot.plot(n_list, [e[2] for e in rt[i]], label="$L^{\infty}$")
    if i in [1]:
        pyplot.plot(*calc_log_line(10.0, 100.0, 0.4, -2.0), linewidth=0.5, color="black")
    if i in [2, 3, 4]:
        pyplot.plot(*calc_log_line(10.0, 100.0, 0.2, -4.0), linewidth=0.5, color="black", linestyle="--")
    pyplot.plot([], linewidth=0.5, color="black", label="Slope -2")
    pyplot.plot([], linewidth=0.5, color="black", linestyle="--", label="Slope -4")
    pyplot.semilogx()
    pyplot.semilogy()
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Figure9.pgf")
pyplot.show()

