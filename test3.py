from scipy import interpolate
#dir(interpolate)[:5]
# print interpolate.spline.__doc__

import numpy as np
from matplotlib import pylab
import seaborn as sns
from CAL.PyCAL import *

font.set_size(20)
x = np.linspace(1.0, 13.0, 6)
y = np.sin(x)
#pylab.figure(figsize = (12,6))
#pylab.scatter(x,y, s = 85, marker='x', color = 'r')
#pylab.title(u'$f(x)$离散点分布', fontproperties = font)

xnew = np.linspace(1.0, 13.0, 10)
ynewLinear = interpolate.spline(x,y,xnew,order = 1)
ynewLinear[:5]

pylab.figure(figsize = (12,6))
pylab.scatter(xnew,ynewLinear, s = 85, marker='x', color = 'r')
pylab.title(u'$f(x)$离散点分布', fontproperties = font)
