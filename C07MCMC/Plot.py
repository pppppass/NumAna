#!/usr/bin/env python
# coding: utf-8

# In[2]:


import shelve
import numpy
import scipy.stats
from matplotlib import pyplot


# In[3]:


db = shelve.open("Result")
res = db["Prob9/Part3/Kin/Final"]
res1 = res


# In[4]:


ns = [16, 32, 64, 128]
ts = [1.0, 1.5, 2.0, 2.5, 3.0]
m = 4
its = numpy.logspace(4.0, 9.0, 21) / m
# its = numpy.logspace(4.0, 7.0, 13) / m
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    for j, t in enumerate(ts):
        ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][l][0][k] for l, _ in enumerate(its)]) for k in [2, 3, 4, 5, 6, 7]]
        mean = ma1
        std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
        pyplot.plot(its, mean, label="$ T = {:.1f} $".format(t), color="C{}".format(j))
        pyplot.scatter(its, mean, s=2.0, color="C{}".format(j))
        pyplot.fill_between(its, mean - eta * std, mean + eta * std, alpha = 0.3, color="C{}".format(j))
#     pyplot.xlabel("$\\mathit{ITER}$")
#     pyplot.ylabel("$m$")
    pyplot.semilogx()
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Figure02.pgf")
pyplot.show()
pyplot.close()


# In[5]:


db.close()


# In[6]:


db = shelve.open("Result")
res = db["Prob9/Part4/Dir/Final"]
res2 = res


# In[7]:


ns = [16, 32, 64, 128]
ts = [1.0, 1.5, 2.0, 2.5, 3.0]
m = 4
its = numpy.logspace(4.0, 9.0, 21) / m
# its = numpy.logspace(4.0, 7.0, 13) / m
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    for j, t in enumerate(ts):
        ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][l][0][k] for l, _ in enumerate(its)]) for k in [2, 3, 4, 5, 6, 7]]
        mean = ma1
        std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
        pyplot.plot(its, mean, label="$ T = {:.1f} $".format(t), color="C{}".format(j))
        pyplot.scatter(its, mean, s=2.0, color="C{}".format(j))
        pyplot.fill_between(its, mean - eta * std, mean + eta * std, alpha = 0.3, color="C{}".format(j))
#     pyplot.xlabel("$\\mathit{ITER}$")
#     pyplot.ylabel("$m$")
    pyplot.semilogx()
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("Figure03.pgf")
pyplot.show()
pyplot.close()


# In[8]:


db.close()


# In[9]:


db = shelve.open("Result")
res = db["Prob1/Part1/Kin/Final"]


# In[10]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(0.5, 4.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = u1
    std = numpy.sqrt((u2 - u1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$u$")
pyplot.tight_layout()
pyplot.savefig("Figure04.pgf")
pyplot.show()
pyplot.close()


# In[12]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(0.5, 4.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = c1
    std = numpy.sqrt((c2 - c1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
#     if i == 3:
#         pyplot.ylim(-0.1, 3.1)
pyplot.tight_layout()
pyplot.savefig("Figure05.pgf")
pyplot.show()
pyplot.close()


# In[13]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(0.5, 4.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = ma1
    std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$m$")
pyplot.tight_layout()
pyplot.savefig("Figure06.pgf")
pyplot.show()
pyplot.close()


# In[16]:


db.close()


# In[17]:


db = shelve.open("Result")
res = db["Prob1/Part2/Kin/Final"]


# In[18]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(2.22, 2.32, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = u1
    std = numpy.sqrt((u2 - u1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$u$")
pyplot.tight_layout()
pyplot.savefig("Figure07.pgf")
pyplot.show()
pyplot.close()


# In[19]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(2.22, 2.32, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = c1
    std = numpy.sqrt((c2 - c1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
pyplot.tight_layout()
pyplot.savefig("Figure08.pgf")
pyplot.show()
pyplot.close()


# In[20]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(2.22, 2.32, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = ma1
    std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
pyplot.tight_layout()
pyplot.savefig("Figure09.pgf")
pyplot.show()
pyplot.close()


# In[21]:


db.close()


# In[22]:


db = shelve.open("Result")
res = db["Prob1/Part1/Dir/Final"]


# In[23]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(0.5, 4.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = u1
    std = numpy.sqrt((u2 - u1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$u$")
pyplot.tight_layout()
pyplot.savefig("Figure10.pgf")
pyplot.show()
pyplot.close()


# In[24]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(0.5, 4.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = c1
    std = numpy.sqrt((c2 - c1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
    if i == 3:
        pyplot.ylim(-0.1, 3.1)
pyplot.tight_layout()
pyplot.savefig("Figure11.pgf")
pyplot.show()
pyplot.close()


# In[25]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(0.5, 4.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = ma1
    std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$m$")
pyplot.tight_layout()
pyplot.savefig("Figure12.pgf")
pyplot.show()
pyplot.close()


# In[26]:


db.close()


# In[27]:


db = shelve.open("Result")
res = db["Prob1/Part2/Dir/Final"]


# In[29]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(2.22, 2.32, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = u1
    std = numpy.sqrt((u2 - u1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$u$")
pyplot.tight_layout()
pyplot.savefig("Figure13.pgf")
pyplot.show()
pyplot.close()


# In[30]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(2.22, 2.32, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = c1
    std = numpy.sqrt((c2 - c1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
pyplot.tight_layout()
pyplot.savefig("Figure14.pgf")
pyplot.show()
pyplot.close()


# In[31]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(2.22, 2.32, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = ma1
    std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
pyplot.tight_layout()
pyplot.savefig("Figure15.pgf")
pyplot.show()
pyplot.close()


# In[32]:


db.close()


# In[33]:


db = shelve.open("Result")
res = db["Prob5/Part1/Kin/Final"]


# In[34]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(2.5, 6.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = u1
    std = numpy.sqrt((u2 - u1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$u$")
pyplot.tight_layout()
pyplot.savefig("Figure16.pgf")
pyplot.show()
pyplot.close()


# In[35]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(2.5, 6.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = c1
    std = numpy.sqrt((c2 - c1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
pyplot.tight_layout()
pyplot.savefig("Figure17.pgf")
pyplot.show()
pyplot.close()


# In[36]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(2.5, 6.5, 41)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = ma1
    std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$m$")
pyplot.tight_layout()
pyplot.savefig("Figure18.pgf")
pyplot.show()
pyplot.close()


# In[37]:


db.close()


# In[38]:


db = shelve.open("Result")
res = db["Prob5/Part2/Kin/Final"]


# In[39]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(4.41, 4.61, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 8:
        ts = numpy.linspace(4.21, 4.41, 21)
    elif n == 12:
        ts = numpy.linspace(4.31, 4.51, 21)
    elif n == 16:
        ts = numpy.linspace(4.34, 4.54, 21)
    elif n == 24:
        ts = numpy.linspace(4.38, 4.58, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = u1
    std = numpy.sqrt((u2 - u1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$u$")
pyplot.tight_layout()
pyplot.savefig("Figure19.pgf")
pyplot.show()
pyplot.close()


# In[40]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(4.41, 4.61, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 8:
        ts = numpy.linspace(4.21, 4.41, 21)
    elif n == 12:
        ts = numpy.linspace(4.31, 4.51, 21)
    elif n == 16:
        ts = numpy.linspace(4.34, 4.54, 21)
    elif n == 24:
        ts = numpy.linspace(4.38, 4.58, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = c1
    std = numpy.sqrt((c2 - c1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
pyplot.tight_layout()
pyplot.savefig("Figure20.pgf")
pyplot.show()
pyplot.close()


# In[41]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(4.41, 4.61, 21)
it = 250000000
m = 4
eta = 3.0
pyplot.figure(figsize=(8.0, 6.0))
for i, n in enumerate(ns):
    if n == 8:
        ts = numpy.linspace(4.21, 4.41, 21)
    elif n == 12:
        ts = numpy.linspace(4.31, 4.51, 21)
    elif n == 16:
        ts = numpy.linspace(4.34, 4.54, 21)
    elif n == 24:
        ts = numpy.linspace(4.38, 4.58, 21)
    pyplot.subplot(2, 2, i+1)
    pyplot.title("$ N = {} $".format(n))
    ma1, ma2, u1, u2, c1, c2 = [numpy.array([res[i][j][k] for j, _ in enumerate(ts)]) for k in [2, 3, 4, 5, 6, 7]]
    mean = ma1
    std = numpy.sqrt((ma2 - ma1**2 + 1.0e-15) / m)
    pyplot.plot(ts, mean, color="C0")
    pyplot.fill_between(ts, mean - eta * std, mean + eta * std, alpha = 0.3, color="C0")
    pyplot.scatter(ts, mean, s=2.0, color="C0")
#     pyplot.xlabel("$T$")
#     pyplot.ylabel("$c$")
pyplot.tight_layout()
pyplot.savefig("Figure21.pgf")
pyplot.show()
pyplot.close()


# In[42]:


db.close()


# In[ ]:




