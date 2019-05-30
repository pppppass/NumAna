#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import intg


# In[2]:


def plot_fig(x, y, z, filename, dpi=200.0):
    pyplot.figure(figsize=(12.0, 4.0))
    pyplot.subplot(1, 3, 1, projection="3d").set_rasterized(True)
    pyplot.gca().scatter(x, y, z, c=numpy.arange(x.size), s=0.3)
    pyplot.gca().plot(x, y, z, linewidth=0.1, c="black")
    pyplot.gca().set_xlabel("$x$")
    pyplot.gca().set_ylabel("$y$")
    pyplot.gca().set_zlabel("$z$")
    pyplot.gca().view_init(azim=-120.0, elev=30.0)
    pyplot.subplot(1, 3, 2, projection="3d").set_rasterized(True)
    pyplot.gca().scatter(x, y, z, c=numpy.arange(x.size), s=0.3)
    pyplot.gca().plot(x, y, z, linewidth=0.1, c="black")
    pyplot.gca().set_xlabel("$x$")
    pyplot.gca().set_ylabel("$y$")
    pyplot.gca().set_zlabel("$z$")
    pyplot.gca().view_init(azim=-60.0, elev=30.0)
    pyplot.subplot(1, 3, 3, projection="3d").set_rasterized(True)
    pyplot.gca().scatter(x, y, z, c=numpy.arange(x.size), s=0.3)
    pyplot.gca().plot(x, y, z, linewidth=0.1, c="black")
    pyplot.gca().set_xlabel("$x$")
    pyplot.gca().set_ylabel("$y$")
    pyplot.gca().set_zlabel("$z$")
    pyplot.gca().view_init(azim=-60.0, elev=-60.0)
    pyplot.tight_layout()
    if filename is not None:
        pyplot.savefig(filename, dpi=dpi)
    pyplot.show()
    pyplot.close()


# In[3]:


n = 50000
h = 1.0e-3
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
x0, y0, z0 = 1.0, 0.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x[:10000], y[:10000], z[:10000], "Figure01.pgf")
plot_fig(x[10000:20000], y[10000:20000], z[10000:20000], "Figure02.pgf")
plot_fig(x[20000:30000], y[20000:30000], z[20000:30000], "Figure03.pgf")
plot_fig(x[30000:40000], y[30000:40000], z[30000:40000], "Figure04.pgf")


# In[4]:


n = 50000
h = 1.0e-3
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
x0, y0, z0 = 0.1, 0.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure05.pgf")
x0, y0, z0 = 1.0, 0.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure06.pgf")
x0, y0, z0 = 10.0, 0.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure07.pgf")
x0, y0, z0 = 100.0, 0.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure08.pgf")


# In[5]:


n = 50000
h = 1.0e-3
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
x0, y0, z0 = 0.1, 0.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure09.pgf")
x0, y0, z0 = 1.0, 0.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure10.pgf")
x0, y0, z0 = 10.0, 0.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure11.pgf")
x0, y0, z0 = 100.0, 0.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure12.pgf")


# In[6]:


n = 50000
h = 1.0e-3
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
x0, y0, z0 = 0.0, 0.1, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure13.pgf")
x0, y0, z0 = 0.0, 1.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure14.pgf")
x0, y0, z0 = 0.0, 10.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure15.pgf")
x0, y0, z0 = 0.0, 100.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure16.pgf")


# In[7]:


n = 50000
h = 1.0e-3
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
x0, y0, z0 = 0.0, 0.1, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure17.pgf")
x0, y0, z0 = 0.0, 1.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure18.pgf")
x0, y0, z0 = 0.0, 10.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure19.pgf")
x0, y0, z0 = 0.0, 100.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure20.pgf")


# In[8]:


n = 50000
h = 1.0e-3
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
x0, y0, z0 = 0.0, 0.0, 0.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure21.pgf")
x0, y0, z0 = 6.0 * numpy.sqrt(2.0), 6.0 * numpy.sqrt(2.0), 27.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure22.pgf")
x0, y0, z0 = -6.0 * numpy.sqrt(2.0), -6.0 * numpy.sqrt(2.0), 27.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure23.pgf")
x0, y0, z0 = 0.0, 0.0, 50.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure24.pgf")


# In[9]:


n = 50000
h = 1.0e-3
x0, y0, z0 = 1.0, 0.0, 0.0
sigma, rho, beta = 10.0, 0.5, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure25.pgf")
sigma, rho, beta = 10.0, 1.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure26.pgf")
sigma, rho, beta = 10.0, 2.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure27.pgf")
sigma, rho, beta = 10.0, 15.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure28.pgf")


# In[10]:


n = 50000
h = 1.0e-3
x0, y0, z0 = 1.0, 0.0, 0.0
sigma, rho, beta = 10.0, 24.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure29.pgf")
sigma, rho, beta = 10.0, 27.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure30.pgf")
sigma, rho, beta = 10.0, 90.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure31.pgf")
sigma, rho, beta = 10.0, 100.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure32.pgf")


# In[11]:


n = 50000
h = 1.0e-3
x0, y0, z0 = 1.0, 0.0, 0.0
sigma, rho, beta = 10.0, 125.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure33.pgf")
sigma, rho, beta = 10.0, 150.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure34.pgf")
sigma, rho, beta = 10.0, 200.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure35.pgf")
sigma, rho, beta = 10.0, 250.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure36.pgf")


# In[12]:


n = 50000
h = 1.0e-3
x0, y0, z0 = 1.0, 0.0, 0.0
sigma, rho, beta = 2.0, 0.5, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure37.pgf")
sigma, rho, beta = 2.0, 2.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure38.pgf")
sigma, rho, beta = 2.0, 20.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure39.pgf")
sigma, rho, beta = 2.0, 200.0, 8.0/3.0
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure40.pgf")


# In[13]:


x0, y0, z0 = 1.0, 0.0, 0.0
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
n = 50000
h = 1.0e-3
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure41.pgf")
x, y, z = intg.intg_ode1_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure42.pgf")
n = 5000
h = 1.0e-2
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure43.pgf")
x, y, z = intg.intg_ode1_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure44.pgf")


# In[14]:


x0, y0, z0 = 1.0, 0.0, 0.0
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
n = 2500
h = 2.0e-2
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure45.pgf")
x, y, z = intg.intg_ode1_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure46.pgf")
n = 1250
h = 4.0e-2
x, y, z = intg.intg_ode4_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure47.pgf")
x, y, z = intg.intg_ode1_lor(n, h, x0, y0, z0, sigma, rho, beta, *(numpy.zeros(n) for _ in range(3)))
plot_fig(x, y, z, "Figure48.pgf")


# In[ ]:




