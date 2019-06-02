#!/usr/bin/env python
# coding: utf-8

# In[15]:


import time
import shelve
import numpy
from matplotlib import pyplot
import samp


# In[16]:


db = shelve.open("Result")


# In[17]:


ns = [16, 32, 64, 128]
ts = [1.0, 1.5, 2.0, 2.5, 3.0]
m = 4
its = numpy.logspace(4.0, 9.0, 21) / m
# its = numpy.logspace(4.0, 7.0, 13) / m


# In[18]:


res = [[[[None for _ in range(2)] for _ in its] for _ in ts] for _ in ns]


# In[19]:


for i, n in enumerate(ns):
    for j, t in enumerate(ts):
        for k, it_float in enumerate(its):
            it = int(it_float + 1.0e-5)
            start = time.time()
            res[i][j][k][0] = samp.driver_samp_ising_kin_2d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
            end = time.time()
            res[i][j][k][1] = end - start
            print("Time: {}".format(end - start))
#             print(res[i][j][k])
            print("it = {} finished".format(it))
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob9/Part3/Kin/Final"] = res
db.sync()


# In[20]:


ns = [16, 32, 64, 128]
ts = [1.0, 1.5, 2.0, 2.5, 3.0]
m = 4
its = numpy.logspace(4.0, 9.0, 21) / m
# its = numpy.logspace(4.0, 7.0, 13) / m


# In[21]:


res = [[[[None for _ in range(2)] for _ in its] for _ in ts] for _ in ns]


# In[23]:


for i, n in enumerate(ns):
    for j, t in enumerate(ts):
        for k, it_float in enumerate(its):
            it = int(it_float + 1.0e-5)
            start = time.time()
            res[i][j][k][0] = samp.driver_samp_ising_2d(n, t, 0.0, it, it//3, m, 65536, numpy.random.randint(0, 2147483647), 4)
            end = time.time()
            res[i][j][k][1] = end - start
            print("Time: {}".format(end - start))
#             print(res[i][j][k])
            print("it = {} finished".format(it))
        print("t = {} finished".format(t))
    print("n = {} finished".format(n))
db["Prob9/Part4/Dir/Final"] = res
db.sync()


# In[ ]:


db.close()

