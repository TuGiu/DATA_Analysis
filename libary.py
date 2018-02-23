#coding=utf-8

from CAL.PyCAL import *
import numpy as np
import math
import seaborn as sns
from matplotlib import pylab
font.set_size(15)
import scipy as sp
from scipy.linalg import solve_banded

class BSMModel:
    def __init__(self, s0, r, sigma):
        self.s0 = s0
        self.x0 = math.log(s0)
        self.r = r
        self.sigma = sigma

    def log_expectation(self, T):
        return self.x0 + (self.r - 0.5 * self.sigma * self.sigma) * T

    def expectation(self, T):
        return self.s0 * math.exp(self.r * T)

    def x_max(self, T):
        return self.log_expectation(T) + 4.0 * self.sigma * math.sqrt(T)

    def x_min(self, T):
        return self.log_expectation(T) - 4.0 * self.sigma * math.sqrt(T) 
    
class CallOption:
    def __init__(self, strike):
        self.k = strike

    def ic(self, spot):
        return max(spot - self.k, 0.0)

    def bcl(self, spot, tau, model):
        return 0.0

    def bcr(self, spot, tau, model):
        return spot - math.exp(-model.r*tau) * self.k 

class BSMThetaScheme:
    def __init__(self, model, payoff, T, M, N, theta, xmin = None, xmax = None):
        self.model = model
        self.T = T
        self.M = M
        self.N = N
        self.dt = self.T / self.M
        
        self.payoff = payoff
        if xmin is None:
            self.x_min = model.x_min(self.T)
        else:
            self.x_min = xmin
            
        if xmax is None:
            self.x_max = model.x_max(self.T)
        else:
            self.x_max = xmax
        self.dx = (self.x_max - self.x_min) / self.N
        self.C = np.zeros((self.N+1, self.M+1)) # 全部网格
        self.xArray = np.linspace(self.x_min, self.x_max, self.N+1)
        self.C[:,0] = map(self.payoff.ic, np.exp(self.xArray))

        sigma_square = self.model.sigma*self.model.sigma
        r = self.model.r
        self.theta = theta

        # Implicit 部分
        self.l_j_implicit = -(1.0 - self.theta) * (0.5*sigma_square*self.dt/self.dx/self.dx - 0.5 * (r - 0.5 * sigma_square)*self.dt/self.dx)
        self.c_j_implicit = 1.0 + (1.0 - self.theta) * (sigma_square*self.dt/self.dx/self.dx + r*self.dt)
        self.u_j_implicit = -(1.0 - self.theta) * (0.5*sigma_square*self.dt/self.dx/self.dx + 0.5 * (r - 0.5 * sigma_square)*self.dt/self.dx)

        # Explicit 部分
        self.l_j_explicit = self.theta * (0.5*sigma_square*self.dt/self.dx/self.dx - 0.5 * (r - 0.5 * sigma_square)*self.dt/self.dx)
        self.c_j_explicit = 1.0 - theta * (sigma_square*self.dt/self.dx/self.dx + r*self.dt)
        self.u_j_explicit = self.theta * (0.5*sigma_square*self.dt/self.dx/self.dx + 0.5 * (r - 0.5 * sigma_square)*self.dt/self.dx)

    def roll_back(self): 

        for k in range(0, self.M):
            rhs = self._apply_explicit(k)
            self._apply_implicti(k,rhs)

    def _apply_explicit(self, k):
        # 应用显示格式部分
        preSol =  np.copy(self.C[:,k])
        rhs = np.array([0.0] * (self.N-1))

        for i in range(self.N-1):
            rhs[i] = self.l_j_explicit * preSol[i] + self.c_j_explicit * preSol[i+1] + self.u_j_explicit * preSol[i+2]
        return rhs

    def _apply_implicti(self, k, rhs):
        # 应用隐式格式部分
        udiag = np.ones(self.N-1) * self.u_j_implicit
        ldiag =  np.ones(self.N-1) * self.l_j_implicit
        cdiag =  np.ones(self.N-1) * self.c_j_implicit

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

        x = solve_banded((1,1), mat, rhs)
        self.C[1:self.N, k+1] = x
        self.C[0][k+1] = v1
        self.C[self.N][k+1] = v2

    def mesh_grids(self):
        tArray = np.linspace(0, self.T, self.M+1)
        tGrids, xGrids = np.meshgrid(tArray, self.xArray)
        return tGrids, xGrids 