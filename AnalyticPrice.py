import numpy as np
import math
import seaborn as sns
from matplotlib import pylab
from CAL.PyCAL import *
font.set_size(15)

import scipy as sp
from scipy.linalg import solve_banded
from lib.PDE import BSMModel, CallOption   # CallOption 买方期权

class BSMThetaScheme:
    def __init__(self, model, payoff, T, M, N, theta):
        self.model = model
        self.T = T
        self.M = M
        self.N = N
        self.dt = self.T / self.M
        self.payoff = payoff
        self.x_min = model.x_min(self.T)
        self.x_max = model.x_max(self.T)
        self.dx = (self.x_max - self.x_min) / self.N
        self.C = np.zeros((self.N+1, self.M+1)) # 全部网格
        self.xArray = np.linspace(self.x_min, self.x_max, self.N+1)
        self.C[:,0] = map(self.payoff.ic, np.exp(self.xArray))
        
        sigma_square = self.model.sigma*self.model.sigma  #sigma ∑
        r = self.model.r
        self.theta = theta
        
        # Implicit 隐性 （部分）
        self.l_j_implicit = -(1.0 - self.theta) * (0.5*sigma_square*self.dt/self.dx/self.dx - 0.5 * (r - 0.5 * sigma_square)*self.dt/self.dx)
        self.c_j_implicit = 1.0 + (1.0 - self.theta) * (sigma_square*self.dx/self.dx/self.dx + r *self.dt)
        self.u_j_implicit = -(1.0 - self.theta) * (0.5*sigma_square*self.dt/self.dx/self.dx + 0.5 * (r - 0.5 * sigma_square)*self.dt/self.dx)
        
        #Explicit (明确的)部分
        self.l_j_explicit = self.theta * (0.5*sigma_square*self.dt/self.dx/self.dx - 0.5 * (r - 0.5 * sigma_square)* self.dt/self.dx)
        self.c_j_explicit = 1.0 - theta * (sigma_square*self.dx/self.dx/self.dx + r*self.dt)
        self.u_j_explicit = self.theta * (0.5*sigma_square*self.dt/self.dx/self.dx + 0.5 * (r - 0.5 * sigma_square)*self.dt/self.dx)
        
    def roll_back(self):
        
        for k in range(0, self.M):
            rhs = self._apply_explicit(k)
            self._apply_implicti(k,rhs)
            
    def _apply_explicit(self, k):    # 应用显示格式部分
        preSol = np.copy(self.C[:,k]) 
        rhs = np.array([0.0] * (self.N-1))
        
        for i in range(self.N-1):
            rhs[i] = self.l_j_explicit * preSol[i] + self.c_j_explicit * preSol[i+1] + self.u_j_explicit * preSol[i+2]
        return rhs
    
    def _apply_implicti(self, k, rhs):  #apply 应用   # 应用隐式格式部分
        udiag = np.ones(self.N-1) * self.u_j_implicit
        ldiag = np.ones(self.N-1) * self.l_j_implicit
        cdiag = np.ones(self.N-1) * self.c_j_implicit
        
        mat = np.zeros((3,self.N-1))
        mat[0,:] = udiag
        mat[1,:] = cdiag
        mat[2,:] = ldiag
        
        # 应用左端边值条件
        v1 = self.payoff.bcl(math.exp(self.x_min), (k+1)*self.dt, self.model)
        rhs[0] -= self.l_j_implicit * v1
        
        # 应用右端边值条件
        v2 = self.payoff.bcr(math.exp(self.x_max), (k+1)*self.dt, self.model)
        rhs[-1] -= self.u_j_implicit * v2
      
model = BSMModel(100.0, 0.05, 0.2)
payoff = CallOption(105.0)
scheme = BSMThetaScheme(model, payoff, 5.0, 100, 300, 0.5)

scheme.roll_back()

from matplotlib import pylab
#pylab.figure(figsize=(12,8))
#pylab.plot(np.exp(scheme.xArray)[50:170], scheme.C[50:170,-1])
#pylab.xlabel('$s$')
#pylab.ylabel('$C$')





# 4. 收敛性测试
analyticPrice = BSMPrice(1, 105., 100., 0.05, 0.0, 0.2, 5.)
analyticPrice

theta = 0.5 # Crank - Nicolson格式
timeStep = 500
xSteps = range(50,300,10)
crankNicolsonResult = [] # crank曲柄 
implicitResult = []
explicitResult = []
for xStep in xSteps:
    model = BSMModel(100.0, 0.05, 0.2)
    payoff = CallOption(105.0)
    scheme = BSMThetaScheme(model, payoff, 5.0, timeStep, xStep, theta)
    scheme.roll_back()
    interp = CubicNaturalSpline(np.exp(scheme.xArray), scheme.C[:,-1])
    price = interp(100.0)
    crankNicolsonResult.append(price)
    
    scheme = BSMThetaScheme(model, payoff, 5.0, timeStep, xStep, 0.0)
    scheme.roll_back()
    interp = CubicNaturalSpline(np.exp(scheme.xArray), scheme.C[:,-1])
    price = interp(100.0)
    implicitResult.append(price)
    

anyRes = [analyticPrice['price'][1]] * len(xSteps)

pylab.figure(figsize = (16,8))
pylab.plot(xSteps, crankNicolsonResult, '-.', marker = 'o', markersize = 10)
pylab.plot(xSteps, implicitResult, '-.', marker = '^', markersize = 10)
pylab.plot(xSteps, anyRes, '--')
pylab.legend([u'Crank - Nicolson差分格式', u'隐式差分格式', u'隐式差分格式', u'解析解（欧式）'], prop = font)
pylab.xlabel(u'价格方向网格步数', fontproperties = font)
pylab.title(u'Black - Scholes - Merton 有限差分法', fontproperties = font, fontsize = 20)