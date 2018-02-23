from CAL.PyCAL import *
from lib.PDE import BSMModel, CallOption, BSMThetaScheme
from matplotlib import pylab
import numpy as np
import math

class PutOption:
    
    def __init__(self, strike):
        self.k = strike
    
    def ic(self, spot):
        return max(self.k - spot, 0.0)
    
    def bcl(self, spot, tau, model):
        return self.k * math.exp(-model.r*tau) - spot
    
    def bcr(self, spot, tau, model):
        return 0.0

analyticPrice = BSMPrice(-1, 105., 100., 0.05, 0.0, 0.2, 5.)
analyticPrice

theta = 0.5 #Crank - Nicolson格式
timeStep = 500
xSteps = range(100,500,10)
crankNicolsonResult = []
for xStep in xSteps:
    model = BSMModel(100.0, 0.05, 0.2)
    payoff = PutOption(105.0)
    scheme = BSMThetaScheme(model, payoff, 5.0, timeStep, xStep, theta)
    scheme.roll_back()
    interp = CubicNaturalSpline(np.exp(scheme.xArray), scheme.C[:,-1])
    price = interp(100.0)
    crankNicolsonResult.append(price)
    
anyRes = [analyticPrice['price'][1]] * len(xSteps)

#pylab.figure(figsize = (16,8))
#pylab.plot(xSteps, crankNicolsonResult, '-.', marker = 'o', markersize = 10)
#pylab.plot(xSteps, anyRes, '--')
#pylab.legend([u'Crank - Nicolson差分格式', u'解析解(欧式)' ], prop = font)
#pylab.xlabel(u'价格方向网格步数', fontproperties = font)
#pylab.title(u'Black - Scholes - Merton 有限差分法（看跌期权）', fontproperties = font, fontsize = 20)



# 二元看涨期权
class DigitialCallOption:
    def __init__(self, strike):
        self.k = strike
        
    def ic(self, spot):
        if spot > self.k:
            return 1.0
        return 0.0
    
    def bcl(self, spot, tau, model):
        return 0.0
    
    def bcr(self, spot, tau, model):
        return 1.0
    
from scipy.stats import norm

def analyticDigitalCall(strike, spot, r, sigma, T):
    d2 = (math.log(spot/strike) + (r - 0.5 * sigma * sigma) * T) / sigma / math.sqrt(T)
    return norm.cdf(d2)*math.exp(-r*T)

res = analyticDigitalCall(105.0, 100.0, 0.05, 0.2, 5.0)
#print res

theta = 0.5
timeStep = 500
xSteps = range(100,2000,50)
crankNicolsonResult = []
for xStep in xSteps:
    model = BSMModel(100.0, 0.05, 0.2)
    payoff = DigitialCallOption(105.0)
    scheme = BSMThetaScheme(model, payoff, 5.0, timeStep, xStep, theta)
    scheme.roll_back()
    interp = CubicNaturalSpline(np.exp(scheme.xArray), scheme.C[:,-1])
    price = interp(100.0)
    crankNicolsonResult.append(price)

anyRes = [res] * len(xSteps)

#pylab.figure(figsize = (16,8))
#pylab.plot(xSteps, crankNicolsonResult, '-.', marker = 'o', markersize = 10)
#pylab.plot(xSteps, anyRes, '--')
#pylab.legend([u'Crank - Nicolson差分格式', u'解析解（欧式）'], prop = font)
#pylab.xlabel(u'价格方向网格步数', fontproperties = font)
#pylab.title(u'二元看涨期权', fontproperties = font, fontsize = 20)


def vanillaCall(strike, spot, r, sigma, T):
    d1 = (math.log(spot/strike) + (r + 0.5 * sigma * sigma) * T) / sigma / math.sqrt(T)
    d2 = (math.log(spot/strike) + (r - 0.5 * sigma * sigma) * T) / sigma / math.sqrt(T)
    return spot * norm.cdf(d1) - strike * math.exp(-r*T) * norm.cdf(d2)

def downOutCall(strike, spot, h, r, sigma, T):
    if h <= 1e-2:
        return vanillaCall(strike, spot, r, sigma, T)
    reflection = h * h / spot
    v = r - 0.5 * sigma * sigma
    callPrice1 = vanillaCall(strike, spot, r, sigma, T)
    callPrice2 = vanillaCall(strike, reflection, r, sigma, T)
    return callPrice1 - ((h/spot) ** (2.0 * v / sigma / sigma)) * callPrice2

res = downOutCall(105.0, 100.0, 95.0, 0.05, 0.2, 5.0)
print res

theta = 0.5
timeStep = 500
xSteps = range(100,500,10)
crankNicolsonResult = []
downBarrier = 95
for xStep in xSteps:
    model = BSMModel(100.0, 0.05, 0.2)
    payoff = CallOption(105.0)
    scheme = BSMThetaScheme(model, payoff, 5.0, timeStep, xStep, theta, xmin = math.log(downBarrier))
    scheme.roll_back()
    interp = CubicNaturalSpline(np.exp(scheme.xArray), scheme.C[:,-1])
    price = interp(100.0)
    crankNicolsonResult.append(price)

anyRes = [res] * len(xSteps)
pylab.figure(figsize = (16,8))
pylab.plot(xSteps, crankNicolsonResult, '-.', marker = 'o', markersize = 10)
pylab.plot(xSteps, anyRes, '--')
pylab.legend([u'Crank - Nicolson差分格式', u'解析解（欧式）'], prop = font)
pylab.xlabel(u'价格方向网格步数', fontproperties = font)
pylab.title(u'向下敲出看涨期权', fontproperties = font, fontsize = 20)
