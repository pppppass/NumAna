#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy
from matplotlib import pyplot
import rand


# In[23]:


def plot_three(x, y, filename):
    
    # Borrowed from https://matplotlib.org/devdocs/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
    
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    pyplot.figure(figsize=(6.0, 6.0))

    ax_scatter = pyplot.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = pyplot.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = pyplot.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    ax_scatter.scatter(x, y)

    binwidth = 0.25
#     lim = numpy.ceil(numpy.abs([x, y]).max() / binwidth) * binwidth
    lim = 3.8
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))

    bins = numpy.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    if filename is not None:
        pyplot.savefig(filename)
    pyplot.show()
    pyplot.close()


# In[24]:


n = 1000


# In[25]:


x, y = rand.gen_rand_gauss_box(n, numpy.zeros(n), numpy.zeros(n), 0)
plot_three(x, y, "Figure22.pgf")


# In[26]:


x, y = rand.gen_rand_gauss_rej(n, numpy.zeros(n), numpy.zeros(n), 0)
plot_three(x, y, "Figure23.pgf")


# In[17]:


n = 10000000
k = 100
t_1, t_2 = 0.0, 0.0
for _ in range(k):
    start = time.time()
    x, y = rand.gen_rand_gauss_box(n, numpy.zeros(n), numpy.zeros(n), 0)
    end = time.time()
    t_1 += end - start
    t_2 += (end - start)**2
print("{:.5f} \pm {:.5f} ".format(t_1 / k, numpy.sqrt(t_2 / k - (t_1 / k)**2)))
with open("Text1.txt", "w") as f:
    f.write("{:.5f} \pm {:.5f} ".format(t_1 / k, numpy.sqrt(t_2 / k - (t_1 / k)**2)))


# In[18]:


n = 10000000
k = 100
t_1, t_2 = 0.0, 0.0
for _ in range(k):
    start = time.time()
    x, y = rand.gen_rand_gauss_rej(n, numpy.zeros(n), numpy.zeros(n), 0)
    end = time.time()
    t_1 += end - start
    t_2 += (end - start)**2
print("{:.5f} \pm {:.5f} ".format(t_1 / k, numpy.sqrt(t_2 / k - (t_1 / k)**2)))
with open("Text2.txt", "w") as f:
    f.write("{:.5f} \pm {:.5f} ".format(t_1 / k, numpy.sqrt(t_2 / k - (t_1 / k)**2)))


# In[ ]:




