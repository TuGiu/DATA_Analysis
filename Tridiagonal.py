from CAL.PyCAL import *
from matplotlib import pylab
import seaborn as sns
import numpy as np

np.set_printoptions(precision = 4)
font.set_size(20)

def initialCondition(x):
    return 4.0*(1.0 - x) * x

N = 500  # x方向网格数
M = 500  # t方向网格数

T = 1.0
X = 1.0

xArray = np.linspace(0,X,N+1)   #(0,1, 500+1)
yArray = map(initialCondition, xArray)

#xArray
#yArray

starValues = yArray
U = np.zeros((N+1, M+1))
U[:,0] = starValues
#print U

dx = X / N
dt = T / M
kappa = 1.0
rho = kappa * dt / dx / dx
#print rho

class TridiagonalSystem:   #Tridiagonal三对角矩阵
    def __init__(self, udiag, cidiag, ldiag):
        '''
        对角矩阵
        udiag -- 上对角线
        cdiag -- 对角线
        ldiag -- 下对角线
        '''
        assert len(udiag) == len(cdiag)  #assert 声称
        assert len(cdiag) == len(ldiag)
        self.udiag = udiag
        self.cdiag = cdiag
        self.ldiag = ldiag
        self.length = len(self.cdiag)  # length长度
        
    def solve(self, rhs):
        '''
        求以下方程组
        A \ dot X = rhs
        '''
        assert len(rhs) == len(self.cdiag)
        udiag = self.udiag.copy()
        cdiag = self.cdiag.copy()
        ldiag = self.ldiag.copy()
        b = rhs.copy()

        #消去下对角元
        for i in range(1, self.length):
            cdiag[i] -= udiag[i-1] * ldiag[i] / cdiag[i-1]
            b[i] -= b[i-1] * ldiag[i] / cdiag[i-1]
            
        
        # 从最后一个方程开始求解
        x = np.zeros(self.length)
        x[self.length-1] = b[self.length -1] / cdiag[self.length -1]
        for i in range(self.length - 2, -1, -1):
            x[i] = (b[i] - udiag[i]*x[i+1]) / cdiag[i]
        return x
    
    def mutiply(self, x):   #mutiply多元化
        '''
        矩阵乘法：
        rhs = A \ dot x
        '''
        assert len(x) == len(self.cdiag)
        rhs = np.zeros(self.length)
        rhs[0] = x[0] * self.cdiag[0] + x[1] * self.udiag[0]
        for i in range(1, self.length - 1):
            rhs[i] = x[i-1] * self.ldiag[i] + x[i] * self.cdiag[i] + x[i+1] * self.udiag[i]
        rhs[self.length - 1] = x[self.length - 2] * self.ldiag[self.length - 1] + x[self.length - 1] * self.cdiag[self.length - 1]
        return rhs

for k in range(0, M):
    udiag = - np.ones(N-1) * rho
    ldiag = - np.ones(N-1) * rho
    cdiag = np.ones(N-1) * (1.0 + 2. * rho)
    
    mat = TridiagonalSystem(udiag, cdiag, ldiag)
    rhs = U[1:N,k]
    x = mat.solve(rhs)
    U[1:N, k+1] = x
    U[0][k+1] = 0.
    U[N][k+1] = 0.

    

from lib.Utilities import plotLines
#plotLines([U[:,0], U[:, int(0.10/ dt)], U[:, int(0.20/ dt)], U[:, int(0.50/ dt)]], xArray, title = u'一维热传导方程', xlabel = '$x$', 
#          ylabel = r'$U(\dot, \tau)$', legend = [r'$\tau = 0.$', r'$\tau = 0.10$', r'$\tau = 0.20$', r'$\tau = 0.50$'])

    

#from lib.Utilities import plotSurface
#tArray = np.linspace(0, 0.2, int(0.2 / dt) + 1)
#tGrids, xGrids = np.meshgrid(tArray, xArray)   # mesh-网格 grid-网格

#plotSurface(xGrids, tGrids, U[:,:int(0.2 / dt) + 1], title = u'热传导方程 $u_\\tau = u_{xx}$, 隐式格式 ($\\rho = 50$)', xlabel = '$x$', ylabel = r'$\tau$', zlabel = r'$U$')


from lib.Utilities import HeatEquation

class ImplicitEulerScheme:
    def __init__(self, M, N, equation):
        self.eq = equation
        self.dt = self.eq.T / M
        self.dx = self.eq.X / N
        self.U = np.zeros((N+1, M+1))
        self.xArray = np.linspace(0,self.eq.X,N+1)
        self.U[:,0] = map(self.eq.ic, self.xArray)
        self.rho = self.eq.kappa * self.dt / self.dx / self.dx
        self.M = M
        self.N = N
    
    def roll_back(self):
        for k in range(0, self.M):
            udiag = - np.ones(self.N-1) * self.rho
            ldiag = - np.ones(self.N-1) * self.rho
            cdiag = np.ones(self.N-1) * (1.0 + 2. * self.rho)
            
            mat = TridiagonalSystem(udiag, cdiag, ldiag)
            rhs = self.U[1:self.N,k]
            x = mat.solve(rhs)
            self.U[1:self.N, k+1] = x
            self.U[0][k+1] = self.eq.bcl(self.xArray[0])
            self.U[self.N][k+1] = self.eq.bcr(self.xArray[-1])
            
        def mesh_grids(self):
            tArray = np.linspace(0, self.eq.T, M+1)
            tGrids, xGrids = np.meshgrid(tArray, self.xArray)
            return tGrids, xGrids

ht = HeatEquation(1., X, T)
#print ht
scheme = ImplicitEulerScheme(M,N, ht)
scheme.roll_back()
#print scheme
#scheme.U

import scipy as sp
from scipy.linalg import solve_banded

A = np.zeros((3, 5))
A[0, :] = np.ones(5) * 1. # 上对角线
A[1, :] = np.ones(5) * 3. # 对角线
A[2, :] = np.ones(5) * (-1.) # 下对角线
            
b = [1.,2.,3.,4.,5.]
x = solve_banded((1,1), A,b)
#print 'x = A^-1b = ',x


import scipy as sp
from scipy.linalg import solve_banded

for k in range(0,M):
    udiag = - np.ones(N-1) * rho
    ldiag = - np.ones(N-1) * rho
    cdiag = np.ones(N-1) * (1.0 + 2. * rho)
    mat = np.zeros((3,N-1))
    mat[0,:] = udiag
    #print mat[0,:]
    mat[1,:] = cdiag
    #print mat[1,:]
    mat[2,:] = ldiag
    #print mat[2,:]
    rhs = U[1:N,k]
    #print rhs
    x = solve_banded ((1,1), mat,rhs)
    U[1:N, k+1] = x
    U[0][k+1] = 0.
    U[N][k+1] = 0.
    
#plotLines([U[:,0], U[:, int(0.1/ dt)], U[:, int(0.20/ dt)], U[:, int(0.50/ dt)]], xArray, title = u'一维热传导方程， 使用scipy', xlabel = '$x$',
#         ylabel = r'$U(\dot, \tau)$', legend = [r'$\tau = 0.$', r'$\tau = 0.10$', r'$\tau = 0.20$', r'$\tau = 0.50$'])


class ImplicitEulerSchemeWithScipy:
    def __init__(self, M, N, equation):
        self.eq = equation
        self.dt = self.eq.T / M
        self.dx = self.eq.X / N
        self.U = np.zeros((N+1, M+1))
        self.xArray = np.linspace(0,self.eq.X,N+1)
        self.U[:,0] = map(self.eq.ic, self.xArray)
        self.rho = self.eq.kappa * self.dt / self.dx / self.dx
        self.M = M
        self.N = N
        
    def roll_back(self):
        for k in range(0, self.M):
            udiag = - np.ones(self.N-1) * self.rho
            ldiag = - np.ones(self.N-1) * self.rho
            cdiag = np.ones(self.N-1) * (1.0 + 2. * self.rho)
            
            mat = np.zeros((3,self.N-1))
            mat[0,:] = udiag
            mat[1,:] = cdiag
            mat[2,:] = ldiag
            rhs = self.U[1:,N,k]
            x = solove_banded((1,1), mat, rhs)
            self.U[1:self.N, k+1] = x
            self.U[0][k+1] = self.eq.bcl(self.xArray[0])
            self.U[self.N][k+1] = self.eq.bcr(self.xArray[-1])
            
        def mesh_grids(self):
            tArray = np.linspace(0, self.eq.T, M+1)
            tGrids, xGrids = np.meshgrid(tArray, self.xArray)
            return tGrids, xGrids


#import time
#startTime = time.time()
#loop_round = 10

# 不使用scipy
#for k in range(loop_round):
    #ht = HeatEquation(1.,X, T)
    #scheme = ImplicitEulerScheme(M,N, ht)
    #scheme.roll_back()
#endTime = time.time()
#print '{0:<40}{1:.4f}'.format('执行时间（s）--不使用scipy.linalg:', endTime - startTime)



# 使用scipy     
startTime = time.time()
for k in range(loop_round):
    ht = HeatEquation(1.,X, T)
    scheme = ImplicitEulerSchemeWithScipy(M,N, ht)
    scheme.roll_back()
endTime = time.time()
print '{0:<40}{1:.4f}'.format('执行时间（s）--使用scipy.linalg: ', endTime - startTime)