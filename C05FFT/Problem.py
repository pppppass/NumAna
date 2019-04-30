#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy
from matplotlib import pyplot
import fft


# In[2]:


def calc_log_line(start, end, intc, order):
    return [start, end], [intc, intc * (end / start)**order]


# In[3]:


ks = [3, 4, 5, 6]


# In[4]:


pyplot.figure(figsize=(8.0, 8.0))
for i, k in enumerate(ks):
    pyplot.subplot(2, 2, i+1)
    n = 2**k
    pyplot.title("$ N = {} $".format(n))
    h = numpy.pi / 3.0 / n
    x = numpy.linspace(0.0, numpy.pi/3.0, n, endpoint=False)
    s = numpy.zeros(n)
    s[1] = 3.0 * n / 2
    u_diff_cen = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 1.0/h, -2.0/h**2 + 2.0, 1.0/h**2 - 1.0/h)
    fft.trans_ifft(k, u_diff_cen)
    pyplot.plot(x, u_diff_cen, label="Central difference")
    u_diff_for = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 2.0/h, -2.0/h**2 - 2.0/h + 2.0, 1.0/h**2)
    fft.trans_ifft(k, u_diff_for)
    pyplot.plot(x, u_diff_for, label="Forward difference")
    u_spec = fft.solve_spec_3(k, s.copy(), 1.0, 2.0, 2.0)
    fft.trans_ifft(k, u_spec)
    pyplot.plot(x, u_spec, label="Spectral")
    u_real = -51.0/650.0 * numpy.cos(6.0*x) + 9.0/325.0 * numpy.sin(6.0*x)
    pyplot.plot(x, u_real, label="Analytical", linewidth=0.5, color="black")
    if i == 3:
        pyplot.legend()
    pyplot.tight_layout()
pyplot.savefig("Figure1.pgf")
pyplot.show()
pyplot.close()


# In[5]:


pyplot.figure(figsize=(8.0, 8.0))
for i, k in enumerate(ks):
    pyplot.subplot(2, 2, i+1)
    n = 2**k
    pyplot.title("$ N = {} $".format(n))
    h = numpy.pi / 3.0 / n
    x = numpy.linspace(0.0, numpy.pi/3.0, n, endpoint=False)
    s = numpy.exp(numpy.sin(6.0*x)) * (2.0 + 12.0 * numpy.cos(6.0*x) + 36.0 * (-numpy.sin(6.0*x) + numpy.cos(6.0*x)**2))
    fft.trans_fft(k, s)
    u_diff_cen = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 1.0/h, -2.0/h**2 + 2.0, 1.0/h**2 - 1.0/h)
    fft.trans_ifft(k, u_diff_cen)
    pyplot.plot(x, u_diff_cen, label="Central difference")
    u_diff_for = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 2.0/h, -2.0/h**2 - 2.0/h + 2.0, 1.0/h**2)
    fft.trans_ifft(k, u_diff_for)
    pyplot.plot(x, u_diff_for, label="Forward difference")
    u_spec = fft.solve_spec_3(k, s.copy(), 1.0, 2.0, 2.0)
    fft.trans_ifft(k, u_spec)
    pyplot.plot(x, u_spec, label="Spectral")
    u_real = numpy.exp(numpy.sin(6.0*x))
    pyplot.plot(x, u_real, label="Analytical", linewidth=0.5, color="black")
    if i == 3:
        pyplot.legend()
    pyplot.tight_layout()
pyplot.savefig("Figure2.pgf")
pyplot.show()
pyplot.close()


# In[6]:


ks = list(range(2, 25))
ns = [2**k for k in ks]


# In[7]:


titles = ["Central difference", "Forward difference", "Spectral"]


# In[8]:


rt = [[], [], []]


# In[9]:


for i, k in enumerate(ks):
    n = 2**k
    h = numpy.pi / 3.0 / n
    x = numpy.linspace(0.0, numpy.pi/3.0, n, endpoint=False)
    start = time.time()
    s = numpy.zeros(n)
    s[1] = 3.0 * n / 2
    end = time.time()
    time_sour = end - start
    u_real = -51.0/650.0 * numpy.cos(6.0*x) + 9.0/325.0 * numpy.sin(6.0*x)
    start = time.time()
    u_diff_cen = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 1.0/h, -2.0/h**2 + 2.0, 1.0/h**2 - 1.0/h)
    fft.trans_ifft(k, u_diff_cen)
    end = time.time()
    rt[0].append((
        end - start + time_sour,
        numpy.linalg.norm(u_diff_cen - u_real, 2.0) * numpy.sqrt(h),
        numpy.linalg.norm(u_diff_cen - u_real, numpy.infty)
    ))
    start = time.time()
    u_diff_for = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 2.0/h, -2.0/h**2 - 2.0/h + 2.0, 1.0/h**2)
    fft.trans_ifft(k, u_diff_for)
    end = time.time()
    rt[1].append((
        end - start + time_sour,
        numpy.linalg.norm(u_diff_for - u_real, 2.0) * numpy.sqrt(h),
        numpy.linalg.norm(u_diff_for - u_real, numpy.infty)
    ))
    start = time.time()
    u_spec = fft.solve_spec_3(k, s.copy(), 1.0, 2.0, 2.0)
    fft.trans_ifft(k, u_spec)
    end = time.time()
    rt[2].append((
        end - start + time_sour,
        numpy.linalg.norm(u_spec - u_real, 2.0) * numpy.sqrt(h),
        numpy.linalg.norm(u_spec - u_real, numpy.infty)
    ))
    print("k = {} finished".format(k))


# In[10]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(3):
    t = numpy.array([e[0] for e in rt[i]])
    pyplot.plot(ns, t, label=titles[i])
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$N$")
    pyplot.ylabel("Time (s)")
pyplot.plot(*calc_log_line(5.0e2, 3.0e7, 1.5e-5, 1.0), linewidth=0.5, color="black", label="Slope $1$")
pyplot.legend()
pyplot.savefig("Figure3.pgf")
pyplot.show()
pyplot.close()


# In[11]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(3):
    e_2 = numpy.array([e[1] for e in rt[i]])
    pyplot.plot(ns, e_2, label=titles[i])
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.plot(*calc_log_line(1.0e1, 3.0e5, 0.3e-1, -1.0), linewidth=0.5, color="black", label="Slope $-1$")
pyplot.plot(*calc_log_line(1.0e1, 3.0e4, 1.0e-2, -2.0), linewidth=0.5, color="black", label="Slope $-2$", linestyle="--")
pyplot.legend()
pyplot.savefig("Figure4.pgf")
pyplot.show()
pyplot.close()


# In[12]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(3):
    e_inf = numpy.array([e[2] for e in rt[i]])
    pyplot.plot(ns, e_inf, label=titles[i])
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.plot(*calc_log_line(1.0e1, 3.0e5, 0.4e-1, -1.0), linewidth=0.5, color="black", label="Slope $-1$")
pyplot.plot(*calc_log_line(1.0e1, 3.0e4, 1.6e-2, -2.0), linewidth=0.5, color="black", label="Slope $-2$", linestyle="--")
pyplot.legend()
pyplot.savefig("Figure5.pgf")
pyplot.show()
pyplot.close()


# In[13]:


rt = [[], [], []]


# In[14]:


for i, k in enumerate(ks):
    n = 2**k
    h = numpy.pi / 3.0 / n
    x = numpy.linspace(0.0, numpy.pi/3.0, n, endpoint=False)
    start = time.time()
    s = numpy.exp(numpy.sin(6.0*x)) * (2.0 + 12.0 * numpy.cos(6.0*x) + 36.0 * (-numpy.sin(6.0*x) + numpy.cos(6.0*x)**2))
    fft.trans_fft(k, s)
    end = time.time()
    time_sour = end - start
    u_real = numpy.exp(numpy.sin(6.0*x))
    start = time.time()
    u_diff_cen = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 1.0/h, -2.0/h**2 + 2.0, 1.0/h**2 - 1.0/h)
    fft.trans_ifft(k, u_diff_cen)
    end = time.time()
    rt[0].append((
        end - start + time_sour,
        numpy.linalg.norm(u_diff_cen - u_real, 2.0) * numpy.sqrt(h),
        numpy.linalg.norm(u_diff_cen - u_real, numpy.infty)
    ))
    start = time.time()
    u_diff_for = fft.solve_diff_3(k, s.copy(), 1.0/h**2 + 2.0/h, -2.0/h**2 - 2.0/h + 2.0, 1.0/h**2)
    fft.trans_ifft(k, u_diff_for)
    end = time.time()
    rt[1].append((
        end - start + time_sour,
        numpy.linalg.norm(u_diff_for - u_real, 2.0) * numpy.sqrt(h),
        numpy.linalg.norm(u_diff_for - u_real, numpy.infty)
    ))
    start = time.time()
    u_spec = fft.solve_spec_3(k, s.copy(), 1.0, 2.0, 2.0)
    fft.trans_ifft(k, u_spec)
    end = time.time()
    rt[2].append((
        end - start + time_sour,
        numpy.linalg.norm(u_spec - u_real, 2.0) * numpy.sqrt(h),
        numpy.linalg.norm(u_spec - u_real, numpy.infty)
    ))
    print("k = {} finished".format(k))


# In[15]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(3):
    t = numpy.array([e[0] for e in rt[i]])
    pyplot.plot(ns, t, label=titles[i])
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$N$")
    pyplot.ylabel("Time (s)")
pyplot.plot(*calc_log_line(3.0e2, 1.0e7, 3.0e-5, 1.0), linewidth=0.5, color="black", label="Slope $1$")
pyplot.legend()
pyplot.savefig("Figure6.pgf")
pyplot.show()
pyplot.close()


# In[16]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(3):
    e_2 = numpy.array([e[1] for e in rt[i]])
    pyplot.plot(ns, e_2, label=titles[i])
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.plot(*calc_log_line(1.0e1, 3.0e5, 3.0e-1, -1.0), linewidth=0.5, color="black", label="Slope $-1$")
pyplot.plot(*calc_log_line(1.0e1, 3.0e4, 1.0e-2, -2.0), linewidth=0.5, color="black", label="Slope $-2$", linestyle="--")
pyplot.legend()
pyplot.savefig("Figure7.pgf")
pyplot.show()
pyplot.close()


# In[17]:


pyplot.figure(figsize=(6.0, 4.0))
for i in range(3):
    e_inf = numpy.array([e[2] for e in rt[i]])
    pyplot.plot(ns, e_inf, label=titles[i])
    pyplot.semilogx()
    pyplot.semilogy()
    pyplot.xlabel("$N$")
    pyplot.ylabel("Error")
pyplot.plot(*calc_log_line(1.0e1, 3.0e5, 5.0e-1, -1.0), linewidth=0.5, color="black", label="Slope $-1$")
pyplot.plot(*calc_log_line(1.0e1, 3.0e4, 1.6e-2, -2.0), linewidth=0.5, color="black", label="Slope $-2$", linestyle="--")
pyplot.legend()
pyplot.savefig("Figure8.pgf")
pyplot.show()
pyplot.close()


# In[ ]:




