# 函数插值--波动率曲面构造
from scipy import interpolate
#dir(interpolate)[:5]
# print interpolate.spline.__doc__

import numpy as np
from matplotlib import pylab
import seaborn as sns
from CAL.PyCAL import *

font.set_size(20)
x = np.linspace(1.0, 13.0, 7)
y = np.sin(x)
#pylab.figure(figsize = (12,6))
# pylab.scatter(x,y, s = 85, marker='x', color = 'r')
# pylab.title(u'$f(x)$离散点分布', fontproperties = font)

xnew = np.linspace(1.0, 13.0, 500)
ynewLinear = interpolate.spline(x,y,xnew,order = 1)
ynewLinear[:5]

ynewCubicSpline = interpolate.spline(x,y,xnew,order = 3)
ynewCubicSpline[:5]

ynewReal = np.sin(xnew)
ynewReal[:5]

#pylab.figure(figsize = (16,8))
#pylab.plot(xnew,ynewReal)
#pylab.plot(xnew,ynewLinear)
#pylab.plot(xnew,ynewCubicSpline)
#pylab.scatter(x,y, s = 160, marker='x', color = 'k')
#pylab.legend([u'真实曲线', u'线性插值', u'样条曲线', u'$f(x)$离散点'], prop = font)
#pylab.title(u'$f(x)$不同插值方法拟合效果：线性插值 v.s 样条插值', fontproperties = font)


# 波动率矩阵（Volatilitie Matrix):
import pandas as pd
pd.options.display.float_format = '{:,>.2f}'.format
dates = [Date(2015,3,25), Date(2015,4,25), Date(2015,6,25), Date(2015,9,25)]
strikes = [2.2, 2.3, 2.4, 2.5, 2.6]
blackVolMatrix = np.array([[0.32562851,  0.29746885,  0.29260648,  0.27679993],
                  [ 0.28841840,  0.29196629,  0.27385023,  0.26511898],
                  [ 0.27659511,  0.27350773,  0.25887604,  0.25283775],
                  [ 0.26969754,  0.25565971,  0.25803327,  0.25407669],
                  [ 0.27773032,  0.24823248,  0.27340796,  0.24814975]])
table = pd.DataFrame(blackVolMatrix * 100, index = strikes, columns = dates, )
table.index.name = u'行权价'
table.columns.name = u'到期时间'
#print u'2015.3.3-10：00波动率矩阵'
#table








# 获取方差矩阵（Variance Matrix):
evaluationDate = Date(2015,3,3)
ttm = np.array([(d - evaluationDate) / 365.0 for d in dates])
varianceMatrix = (blackVolMatrix**2) * ttm
varianceMatrix

interp = interpolate.interp2d(ttm, strikes, varianceMatrix, kind = 'linear')
interp(ttm[0], strikes[0])

sMeshes = np.linspace(strikes[0], strikes[-1], 400)
tMeshes = np.linspace(ttm[0], ttm[-1], 200)
interpolatedVarianceSurface = np.zeros((len(sMeshes), len(tMeshes)))
for i, s in enumerate(sMeshes):
    for j, t in enumerate(tMeshes):
        interpolatedVarianceSurface[i][j] = interp(t,s)
        
interpolatedVolatilitySurface = np.sqrt((interpolatedVarianceSurface / tMeshes))
print u'新权价方向网格数:', np.size(interpolatedVolatilitySurface, 0)
print u'到期时间方向网格数：', np.size(interpolatedVarianceSurface, 1)

#pylab.figure(figsize = (16,8))
#pylab.plot(sMeshes, interpolatedVarianceSurface[:, 0])
#pylab.scatter(x = strikes, y = blackVolMatrix[:,0], s = 160,marker = 'x', color = 'r')
#pylab.legend([u'波动率（线性插值）', u'波动率（离散）'], prop = font)
#pylab.title(u'到期时间为2015.3.25期权波动率', fontproperties = font)





from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

maturityMesher, strikeMesher = np.meshgrid(tMeshes, sMeshes)
pylab.figure(figsize =(16,9))
ax = pylab.gca(projection = '3d')
surface = ax.plot_surface(strikeMesher, maturityMesher, interpolatedVolatilitySurface*100, cmap = cm.jet)
pylab.colorbar(surface,shrink=0.75)
pylab.title(u'', fontproperties = font)
pylab.xlabel('strike')
pylab.ylabel("maturity")
ax.set_zlabel(r"volatility(%)")