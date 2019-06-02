#!/usr/bin/env python
# coding: utf-8

# In[13]:


import time
import shelve
import numpy
from matplotlib import pyplot
import samp


# In[14]:


db = shelve.open("Result")


# In[15]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(0.5, 4.5, 41)
it = 250000000
m = 4


# In[16]:


res = [[None for _ in ts] for _ in ns]


# In[17]:


for i, n in enumerate(ns):
    for j, t in enumerate(ts):
        start = time.time()
        res[i][j] = samp.driver_samp_ising_kin_2d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
        end = time.time()
        print("Time: {}".format(end - start))
#         print(res[i][j])
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob1/Part1/Kin/Final"] = res
db.sync()


# In[18]:


res = [[None for _ in ts] for _ in ns]


# In[19]:


for i, n in enumerate(ns):
    for j, t in enumerate(ts):
        start = time.time()
        res[i][j] = samp.driver_samp_ising_2d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
        end = time.time()
        print("Time: {}".format(end - start))
#         print(res[i][j])
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob1/Part1/Dir/Final"] = res
db.sync()


# In[20]:


ns = [16, 32, 64, 128]
ts = numpy.linspace(2.22, 2.32, 21)
it = 250000000
m = 4


# In[21]:


res = [[None for _ in ts] for _ in ns]


# In[22]:


for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    for j, t in enumerate(ts):
        start = time.time()
        res[i][j] = samp.driver_samp_ising_kin_2d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
        end = time.time()
        print("Time: {}".format(end - start))
#         print(res[i][j])
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob1/Part2/Kin/Final"] = res
db.sync()


# In[23]:


res = [[None for _ in ts] for _ in ns]


# In[24]:


for i, n in enumerate(ns):
    if n == 16:
        ts = numpy.linspace(2.22, 2.42, 21)
    elif n == 32:
        ts = numpy.linspace(2.19, 2.39, 21)
    elif n == 64:
        ts = numpy.linspace(2.18, 2.38, 21)
    elif n == 128:
        ts = numpy.linspace(2.17, 2.37, 21)
    for j, t in enumerate(ts):
        start = time.time()
        res[i][j] = samp.driver_samp_ising_2d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
        end = time.time()
        print("Time: {}".format(end - start))
#         print(res[i][j])
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob1/Part2/Dir/Final"] = res
db.sync()


# In[25]:


db.close()

