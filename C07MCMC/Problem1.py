#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy
from matplotlib import pyplot
import samp


# In[16]:


n = 32
ts = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
it = 100000000
m = 8


# In[17]:


q = numpy.zeros((len(ts), m, n, n), dtype=numpy.int32)


# In[18]:


for i, t in enumerate(ts):
    samp.driver_samp_ising_single_2d(n, t, 0.0, it, m, 65536, numpy.random.randint(0, 2147483647), 4, q[i])
    print("t = {} finished".format(t))


# In[19]:


pyplot.figure(figsize=(8.0, 11.0))
for i, t in enumerate(ts):
    for j in range(m):
        pyplot.subplot(m, len(ts), j*len(ts) + i + 1)
        if j == 0:
            pyplot.title("$ T = {} $".format(t))
        pyplot.imshow(q[i, j])
        pyplot.axis("off")
# pyplot.tight_layout()
pyplot.savefig("Figure01.pgf")
pyplot.show()
pyplot.close()


# In[ ]:




