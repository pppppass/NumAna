#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import shelve
import numpy
from matplotlib import pyplot
import samp


# In[2]:


db = shelve.open("Result")


# In[3]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(2.5, 6.5, 41)
it = 250000000
m = 4


# In[4]:


res = [[None for _ in ts] for _ in ns]


# In[6]:


for i, n in enumerate(ns):
    for j, t in enumerate(ts):
        start = time.time()
        res[i][j] = samp.driver_samp_ising_kin_3d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
        end = time.time()
        print("Time: {}".format(end - start))
#         print(res[i][j])
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob5/Part1/Kin/Final"] = res
db.sync()


# In[14]:


ns = [8, 12, 16, 24]
ts = numpy.linspace(4.41, 4.61, 21)
it = 250000000
m = 4


# In[15]:


res = [[None for _ in ts] for _ in ns]


# In[16]:


for i, n in enumerate(ns):
    if n == 8:
        ts = numpy.linspace(4.21, 4.41, 21)
    elif n == 12:
        ts = numpy.linspace(4.31, 4.51, 21)
    elif n == 16:
        ts = numpy.linspace(4.34, 4.54, 21)
    elif n == 24:
        ts = numpy.linspace(4.38, 4.58, 21)
    for j, t in enumerate(ts):
        start = time.time()
        res[i][j] = samp.driver_samp_ising_kin_3d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
        end = time.time()
        print("Time: {}".format(end - start))
#         print(res[i][j])
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob5/Part2/Kin/Final"] = res
db.sync()


# In[ ]:


db.close()

