#!/usr/bin/env python
# coding: utf-8

# In[18]:


import time
import numpy
from matplotlib import pyplot
import quad


# In[19]:


def calc_log_line(start, end, intc, order):
    return [start, end], [intc, intc * (end / start)**order]


# In[20]:


def clean_sharp(arr):
    arr = numpy.array(arr)
    arr[arr < 1.0e-16] = numpy.nan
    return arr


# In[21]:


i_ana = numpy.pi**4 / 15.0


# In[22]:


eps = 3.0e-6
omega = 50.0
delta = 0.02
f = lambda x: numpy.where(
    x > eps, x**3 / (numpy.exp(x) - 1.0), x**2 / numpy.exp(x / 2.0)
)
f_exp = lambda x: numpy.where(
    x > eps, numpy.where(
        x < omega, x**3 * numpy.exp(x) / (numpy.exp(x) - 1.0), x**3 / (1.0 - numpy.exp(-x))
    ), x**2 * numpy.exp(x / 2.0))
x_high = 100.0
g = lambda y: numpy.where(
    y < 1.0 - eps, numpy.where(
        y > delta, (1.0 / y - 1.0)**3 * numpy.e / y**2 / (numpy.exp(1.0 / y) - numpy.e), (1.0 - y)**3 * numpy.e / (numpy.exp((1.0 + 5.0 * y * numpy.log(y + 2.0e-16)) / y) - y**5 * numpy.e)
    ), (1.0 / y - 1.0)**2 / y**2 / numpy.exp(1.0 / 2.0 / y - 1.0 / 2.0)
)


# In[6]:


x = numpy.linspace(0.0, 20.0, 1000)
y = f(x)
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(x, y)
pyplot.xlabel("$x$")
pyplot.ylabel("$f$")
pyplot.savefig("Figure1.pgf")
pyplot.show()


# In[7]:


x = numpy.linspace(0.0, 1.0, 1000)
y = g(x)
pyplot.figure(figsize=(6.0, 4.0))
pyplot.plot(x, y)
pyplot.xlabel("$y$")
pyplot.ylabel("$g$")
pyplot.savefig("Figure2.pgf")
pyplot.show()


# In[144]:


print([int(1.0 * 2**(k/3) + 0.5) for k in range(100)])


# In[145]:


n_list_1 = [
    2, 3, 4, 5, 6, 8, 10, 13, 16,
    20, 25, 32, 40, 51, 64,
    81, 102, 128, 161, 203, 256,
    323, 406, 512, 645, 813, 1024,
    1290, 1625, 2048, 2580, 3251, 4096,
    5161, 6502, 8192, 10321, 13004, 16384,
    20643, 26008, 32768, 41285, 52016, 65536,
    82570, 104032, 131072, 165140, 208064, 262144
]
rep = 100
rt_1 = [[], [], [], [], [], []]


# In[146]:


for n in n_list_1:
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0 + x_high / 2.0 / n, x_high - x_high / 2.0 / n, n)
        y = f(x)
        i_mid = quad.quad_mid_unif(n, 0.0, x_high, y)
    end = time.time()
    rt_1[0].append((i_mid, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, x_high, n+1)
        y = f(x)
        i_trap = quad.quad_trap_unif(n, 0.0, x_high, y)
    end = time.time()
    rt_1[1].append((i_trap, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, x_high, 2*n+1)
        y = f(x)
        i_simp = quad.quad_simp_unif(n, 0.0, x_high, y)
    end = time.time()
    rt_1[2].append((i_simp, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, x_high, 4*n+1)
        y = f(x)
        i_romb = quad.quad_romb_unif(n, 2, 0.0, x_high, y)
    end = time.time()
    rt_1[3].append((i_romb, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, x_high, 8*n+1)
        y = f(x)
        i_romb = quad.quad_romb_unif(n, 3, 0.0, x_high, y)
    end = time.time()
    rt_1[4].append((i_romb, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, x_high, 16*n+1)
        y = f(x)
        i_romb = quad.quad_romb_unif(n, 4, 0.0, x_high, y)
    end = time.time()
    rt_1[5].append((i_romb, (end - start) / rep, x.size))
    
    print("n = {} finished".format(n))


# In[147]:


n_list_2 = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16,
    18, 20, 23, 25, 29, 32, 36, 40, 45, 51, 57, 64,
    72, 81, 91, 102, 114, 128, 144, 161, 181, 203, 228, 256,
    287, 323, 362, 406, 456, 512, 575, 645, 724, 813, 912, 1024,
    1149, 1290, 1448, 1625, 1825, 2048, 2299, 2580, 2896, 3251, 3649, 4096
]
rep = 100
rt_2 = [[], []]


# In[148]:


for n in n_list_2:
    
    start = time.time()
    x, w = quad.calc_lagu_para(n, numpy(n), numpy(n))
    for r in range(rep):
        y = f_exp(x)
        i_lagu = y.dot(w)
    end = time.time()
    rt_2[0].append((i_lagu, (end - start) / rep, x.size))
    
    start = time.time()
    x, w = quad.calc_lege_para(n, 0.0, x_high, numpy(n), numpy(n))
    for r in range(rep):
        y = f(x)
        i_lege = y.dot(w)
    end = time.time()
    rt_2[1].append((i_lege, (end - start) / rep, x.size))
    
    print("n = {} finished".format(n))


# In[155]:


titles_1 = ["Midpoint", "Trapezoid", "Simpson", "Romberg $ k = 2 $", "Romberg $ k = 3 $", "Romberg $ k = 4 $"]
titles_2 = ["Laguerre", "Legendre"]
pyplot.figure(figsize=(8.0, 6.0))
for i in range(6):
    err_list = [numpy.abs(res[0] - i_ana) for res in rt_1[i]]
    pyplot.plot(n_list_1, clean_sharp(err_list), label=titles_1[i])
    pyplot.scatter(n_list_1, clean_sharp(err_list), s=2.0)
for i in range(2):
    err_list= [numpy.abs(res[0] - i_ana) for res in rt_2[i]]
    pyplot.plot(n_list_2, clean_sharp(err_list), label=titles_2[i])
    pyplot.scatter(n_list_2, clean_sharp(err_list), s=2.0)
pyplot.plot(*calc_log_line(30.0, 3.0e4, 2.0, -4.0), linewidth=0.5, color="black", label="Slope $-4$")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$n$")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.savefig("Figure3.pgf")
pyplot.show()


# In[168]:


titles_1 = ["Midpoint", "Trapezoid", "Simpson", "Romberg $ k = 2 $", "Romberg $ k = 3 $", "Romberg $ k = 4 $"]
titles_2 = ["Laguerre", "Legendre"]
pyplot.figure(figsize=(8.0, 6.0))
for i in range(6):
    time_list = [res[1] for res in rt_1[i]]
    pyplot.plot(n_list_1, time_list, label=titles_1[i])
    pyplot.scatter(n_list_1, time_list, s=2.0)
for i in range(2):
    time_list = [res[1] for res in rt_2[i]]
    pyplot.plot(n_list_2, time_list, label=titles_2[i])
    pyplot.scatter(n_list_2, time_list, s=2.0)
pyplot.plot(*calc_log_line(2.0e3, 2.0e5, 1.0e-4, 1.0), linewidth=0.5, color="black", label="Slope $1$")
pyplot.plot(*calc_log_line(5.0e2, 5.0e3, 3.0e-5, 2.0), linewidth=0.5, color="black", label="Slope $2$", linestyle="--")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$n$")
pyplot.ylabel("Time (s)")
pyplot.legend()
pyplot.savefig("Figure4.pgf")
pyplot.show()


# In[170]:


titles_1 = ["Midpoint", "Trapezoid", "Simpson", "Romberg $ k = 2 $", "Romberg $ k = 3 $", "Romberg $ k = 4 $"]
titles_2 = ["Laguerre", "Legendre"]
pyplot.figure(figsize=(8.0, 6.0))
for i in range(6):
    err_list = [numpy.abs(res[0] - i_ana) for res in rt_1[i]]
    time_list = [res[1] for res in rt_1[i]]
    pyplot.plot(time_list, clean_sharp(err_list), label=titles_1[i])
    pyplot.scatter(time_list, clean_sharp(err_list), s=2.0)
for i in range(2):
    err_list = [numpy.abs(res[0] - i_ana) for res in rt_2[i]]
    time_list = [res[1] for res in rt_2[i]]
    pyplot.plot(time_list, clean_sharp(err_list), label=titles_2[i])
    pyplot.scatter(time_list, clean_sharp(err_list), s=2.0)
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("Time (s)")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.savefig("Figure5.pgf")
pyplot.show()


# In[23]:


n_list_1 = [
    2, 3, 4, 5, 6, 8, 10, 13, 16,
    20, 25, 32, 40, 51, 64,
    81, 102, 128, 161, 203, 256,
    323, 406, 512, 645, 813, 1024,
    1290, 1625, 2048, 2580, 3251, 4096,
    5161, 6502, 8192, 10321, 13004, 16384,
]
rep = 100
rt_1 = [[], [], [], [], [], []]


# In[24]:


for n in n_list_1:
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0 + 1.0 / 2.0 / n, 1.0 - 1.0 / 2.0 / n, n)
        y = g(x)
        i_mid = quad.quad_mid_unif(n, 0.0, 1.0, y)
    end = time.time()
    rt_1[0].append((i_mid, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, 1.0, n+1)
        y = g(x)
        i_trap = quad.quad_trap_unif(n, 0.0, 1.0, y)
    end = time.time()
    rt_1[1].append((i_trap, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, 1.0, 2*n+1)
        y = g(x)
        i_simp = quad.quad_simp_unif(n, 0.0, 1.0, y)
    end = time.time()
    rt_1[2].append((i_simp, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, 1.0, 4*n+1)
        y = g(x)
        i_romb = quad.quad_romb_unif(n, 2, 0.0, 1.0, y)
    end = time.time()
    rt_1[3].append((i_romb, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, 1.0, 8*n+1)
        y = g(x)
        i_romb = quad.quad_romb_unif(n, 3, 0.0, 1.0, y)
    end = time.time()
    rt_1[4].append((i_romb, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(0.0, 1.0, 16*n+1)
        y = g(x)
        i_romb = quad.quad_romb_unif(n, 4, 0.0, 1.0, y)
    end = time.time()
    rt_1[5].append((i_romb, (end - start) / rep, x.size))
    
    print("n = {} finished".format(n))


# In[25]:


n_list_2 = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16,
    18, 20, 23, 25, 29, 32, 36, 40, 45, 51, 57, 64,
    72, 81, 91, 102, 114, 128, 144, 161, 181, 203, 228, 256,
    287, 323, 362, 406, 456, 512, 575, 645, 724, 813, 912, 1024,
    1149, 1290, 1448, 1625, 1825, 2048, 2299, 2580, 2896, 3251, 3649, 4096
]
rep = 100
rt_2 = [[]]


# In[26]:


for n in n_list_2:
    
    start = time.time()
    x, w = quad.calc_lege_para(n, 0.0, 1.0, numpy.zeros(n), numpy.zeros(n))
    for r in range(rep):
        y = g(x)
        i_lege = y.dot(w)
    end = time.time()
    rt_2[0].append((i_lege, (end - start) / rep, x.size))
    
    print("n = {} finished".format(n))


# In[27]:


titles_1 = ["Midpoint", "Trapezoid", "Simpson", "Romberg $ k = 2 $", "Romberg $ k = 3 $", "Romberg $ k = 4 $"]
titles_2 = ["Legendre"]
pyplot.figure(figsize=(8.0, 6.0))
for i in range(6):
    err_list = [numpy.abs(res[0] - i_ana) for res in rt_1[i]]
    pyplot.plot(n_list_1, clean_sharp(err_list), label=titles_1[i])
    pyplot.scatter(n_list_1, clean_sharp(err_list), s=2.0)
for i in range(1):
    err_list = [numpy.abs(res[0] - i_ana) for res in rt_2[i]]
    pyplot.plot(n_list_2, clean_sharp(err_list), label=titles_2[i])
    pyplot.scatter(n_list_2, clean_sharp(err_list), s=2.0)
pyplot.plot(*calc_log_line(1.0e2, 1.0e3, 6.0e-10, -4.0), linewidth=0.5, color="black", label="Slope $-4$")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$n$")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.savefig("Figure6.pgf")
pyplot.show()


# In[28]:


titles_1 = ["Midpoint", "Trapezoid", "Simpson", "Romberg $ k = 2 $", "Romberg $ k = 3 $", "Romberg $ k = 4 $"]
titles_2 = ["Legendre"]
pyplot.figure(figsize=(8.0, 6.0))
for i in range(6):
    time_list = [res[1] for res in rt_1[i]]
    pyplot.plot(n_list_1, time_list, label=titles_1[i])
    pyplot.scatter(n_list_1, time_list, s=2.0)
for i in range(1):
    time_list = [res[1] for res in rt_2[i]]
    pyplot.plot(n_list_2, time_list, label=titles_2[i])
    pyplot.scatter(n_list_2, time_list, s=2.0)
pyplot.plot(*calc_log_line(5.0e2, 1.0e4, 0.7e-4, 1.0), linewidth=0.5, color="black", label="Slope $1$")
pyplot.plot(*calc_log_line(1.0e3, 5.0e3, 2.0e-4, 2.0), linewidth=0.5, color="black", label="Slope $2$", linestyle="--")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$n$")
pyplot.ylabel("Time (s)")
pyplot.legend()
pyplot.savefig("Figure7.pgf")
pyplot.show()


# In[29]:


titles_1 = ["Midpoint", "Trapezoid", "Simpson", "Romberg $ k = 2 $", "Romberg $ k = 3 $", "Romberg $ k = 4 $"]
titles_2 = ["Legendre"]
pyplot.figure(figsize=(8.0, 6.0))
for i in range(6):
    err_list = [numpy.abs(res[0] - i_ana) for res in rt_1[i]]
    time_list = [res[1] for res in rt_1[i]]
    pyplot.plot(time_list, clean_sharp(err_list), label=titles_1[i])
    pyplot.scatter(time_list, clean_sharp(err_list), s=2.0)
for i in range(1):
    err_list = [numpy.abs(res[0] - i_ana) for res in rt_2[i]]
    time_list = [res[1] for res in rt_2[i]]
    pyplot.plot(time_list, clean_sharp(err_list), label=titles_2[i])
    pyplot.scatter(time_list, clean_sharp(err_list), s=2.0)
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("Time (s)")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.savefig("Figure8.pgf")
pyplot.show()


# In[187]:


x_low, x_high = 2.5, 7.5
# n = 128
# x, w = quad.calc_lege_para(n, x_low, x_high, numpy.zeros(n), numpy.zeros(n))
# y = f(x)
# i_ref = y.dot(w)


# In[188]:


n = 1024
i_ref = 0.0
p = lambda t: (t**3 + 3.0 * t**2 + 6.0 * t + 6.0) * numpy.exp(-t)
for k in range(1, n):
    i_ref += (p(x_low * k) - p(x_high * k)) / k**4


# In[189]:


n_list_1 = [
    2, 3, 4, 5, 6, 8, 10, 13, 16,
    20, 25, 32, 40, 51, 64,
    81, 102, 128, 161, 203, 256,
    323, 406, 512, 645, 813, 1024,
    1290, 1625, 2048, 2580, 3251, 4096,
    5161, 6502, 8192, 10321, 13004, 16384,
    20643, 26008, 32768, 41285, 52016, 65536,
    82570, 104032, 131072, 165140, 208064, 262144,
    330281, 416128, 524288, 660561, 832255, 1048576,
]
rep = 1
rt_1 = [[], [], [], [], [], []]


# In[190]:


for n in n_list_1:
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(x_low + (x_high - x_low) / 2.0 / n, x_high - (x_high - x_low) / 2.0 / n, n)
        y = f(x)
        i_mid = quad.quad_mid_unif(n, x_low, x_high, y)
    end = time.time()
    rt_1[0].append((i_mid, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(x_low, x_high, n+1)
        y = f(x)
        i_trap = quad.quad_trap_unif(n, x_low, x_high, y)
    end = time.time()
    rt_1[1].append((i_trap, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(x_low, x_high, 2*n+1)
        y = f(x)
        i_simp = quad.quad_simp_unif(n, x_low, x_high, y)
    end = time.time()
    rt_1[2].append((i_simp, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(x_low, x_high, 4*n+1)
        y = f(x)
        i_romb = quad.quad_romb_unif(n, 2, x_low, x_high, y)
    end = time.time()
    rt_1[3].append((i_romb, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(x_low, x_high, 8*n+1)
        y = f(x)
        i_romb = quad.quad_romb_unif(n, 3, x_low, x_high, y)
    end = time.time()
    rt_1[4].append((i_romb, (end - start) / rep, x.size))
    
    start = time.time()
    for r in range(rep):
        x = numpy.linspace(x_low, x_high, 16*n+1)
        y = f(x)
        i_romb = quad.quad_romb_unif(n, 4, x_low, x_high, y)
    end = time.time()
    rt_1[5].append((i_romb, (end - start) / rep, x.size))
    
    print("n = {} finished".format(n))


# In[200]:


titles_1 = ["Midpoint", "Trapezoid", "Simpson", "Romberg $ k = 2 $", "Romberg $ k = 3 $", "Romberg $ k = 4 $"]
pyplot.figure(figsize=(8.0, 6.0))
for i in range(6):
    err_list = [numpy.abs(res[0] - i_ref) for res in rt_1[i]]
    pyplot.plot(n_list_1, clean_sharp(err_list), label=titles_1[i])
    pyplot.scatter(n_list_1, clean_sharp(err_list), s=2.0)
pyplot.plot(*calc_log_line(2.0, 1.0e6, 0.6, -2.0), linewidth=0.5, color="black", label="Slope $-2$")
pyplot.plot(*calc_log_line(2.0, 1.0e3, 1.2e-2, -4.0), linewidth=0.5, color="black", label="Slope $-4$", linestyle="--")
pyplot.plot(*calc_log_line(2.0, 0.8e2, 2.0e-4, -6.0), linewidth=0.5, color="black", label="Slope $-6$", linestyle=":")
pyplot.plot(*calc_log_line(3.0, 15.0, 3.0e-9, -8.0), linewidth=0.5, color="black", label="Slope $-8$", linestyle="-.")
pyplot.semilogx()
pyplot.semilogy()
pyplot.xlabel("$n$")
pyplot.ylabel("Error")
pyplot.legend()
pyplot.savefig("Figure9.pgf")
pyplot.show()


# In[ ]:




