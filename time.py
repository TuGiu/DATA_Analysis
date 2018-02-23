spot = 2.45         #当前价
strike = 2.50       #行权价
maturity = 0.25     #到期期限
r = 0.05            #无风险利率
vol = 0.25          #波动率

from math import log, sqrt, exp
from scipy.stats import norm

def call_option_pricer(spot, strike, matturity, r, vol):
    
    d1 = (log(spot/strike) + (r + 0.5 * vol *vol) * maturity) / vol / sqrt(maturity)
    d2 = d1 - vol * sqrt(maturity)
    
    price = spot * norm.cdf(d1) - strike * exp(-r*maturity) * norm.cdf(d2)
    return price

#print '期权价格 ： %.4f' % call_option_pricer(spot, strike, maturity, r, vol)


import time
import numpy as np

portfolioSize = range(1, 10000, 500)
timeSpent = []
# print portfolioSize

for size in portfolioSize:
    now = time.time()
    strikes = np.linspace(2.0,3.0,size)
#print strikes
    for i in range(size):
        res = call_option_pricer(spot, strikes[i], maturity, r, vol)
    timeSpent.append(time.time() - now)
    
from matplotlib import pylab
import seaborn as sns
from CAL.PyCAL import *

#font.set_size(15)
#sns.set(style="ticks")
#pylab.figure(figsize = (12,8))
#pylab.bar(portfolioSize, timeSpent, color = 'b', width=300)  #绘制
# pylab.grid(True)    #格子
# pylab.title(u'期权计算时间耗时（单位：秒)', fontproperties = font, fontsize = 18)
# pylab.ylabel(u'时间(s)', fontproperties = font, fontsize = 15)
# pylab.xlabel(u'组合数量', fontproperties = font, fontsize = 15)

sample = np.linspace(1.0, 100.0, 5)
print sample
np.exp(sample)
    
# 使用numpy的向量函数重写Black - Scholes公式
def call_option_pricer_nunmpy(spot, strike, maturity, r, vol):
    
    d1 = (np.log(spot/strike) + (r + 0.5 * vol *vol) * maturity) / vol / np.sqrt(maturity)
    d2 = d1 - vol * np.sqrt(maturity)
    
    price = spot * norm.cdf(d1) - strike * np.exp(-r*maturity) * norm.cdf(d2)
    return price

timeSpentNumpy = []
for size in portfolioSize:
    now = time.time()
    strikes = np.linspace(2.0,3.0, size)
    res = call_option_pricer_nunmpy(spot, strikes, maturity, r, vol)
    timeSpentNumpy.append(time.time() - now)
    
#pylab.figure(figsize = (12,8))
#pylab.bar(portfolioSize, timeSpentNumpy, color = 'r', width = 300)
#pylab.grid(True)   #grid格子
#pylab.title(u'期权计算时间耗时（单位：秒）-numpy加速版', fontproperties = font, fontsize = 18)
#pylab.ylabel(u'时间(s)', fontproperties = font, fontsize = 15)
#pylab.xlabel(u'组合数量', fontproperties = font, fontsize = 15)




    
# 把两次计算时间进行比对
fig = pylab.figure(figsize = (12, 8))   #figure图形
ax = fig.gca()
pylab.plot(portfolioSize, np.log10(timeSpent), portfolioSize, np.log(timeSpentNumpy))
pylab.grid(True)
from matplotlib.ticker import FuncFormatter
def millions(x, pos):
    'The two args are the value and tick position'
    return '$10^{%.0f}$' % (x)
formatter = FuncFormatter(millions)
ax.yaxis.set_major_formatter(formatter)
pylab.title(u'期权计算时间耗时（单位：秒）', fontproperties = font, fontsize = 18)
pylab.legend([u'循环计算',u'numpy向量加速'], prop = font, loc = 'upper center', ncol = 2)  #legend图例
pylab.ylabel(u'时间（秒）', fontproperties = font, fontsize = 15)
pylab.xlabel(u'组合数量', fontproperties = font, fontsize = 15)
print formatter