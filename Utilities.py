#coding=utf-8

from CAL.PyCAL import *
from matplotlib import pylab
import seaborn as sns
import numpy as np
font.set_size(20)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class HeatEquation:    
    def __init__(self, kappa, X, T,
             initialConstion = lambda x:4.0*x*(1.0-x), boundaryConditionL = lambda x: 0, boundaryCondtionR = lambda x:0):
        self.kappa = kappa
        self.ic = initialConstion
        self.bcl = boundaryConditionL
        self.bcr = boundaryCondtionR
        self.X = X
        self.T = T
        
def plotSurface(xGrids, yGrids, zGrids, title, xlabel, ylabel, zlabel):
    fig= pylab.figure(figsize = (16,10))
    ax = fig.add_subplot(1, 1, 1, projection = '3d')
    surface = ax.plot_surface(xGrids, yGrids, zGrids, cmap=cm.coolwarm)
    ax.set_xlabel(xlabel, fontdict={"size":18})
    ax.set_ylabel(ylabel, fontdict={"size":18})
    ax.set_zlabel(zlabel, fontdict={"size":18})
    ax.set_title(title , fontproperties = font)
    fig.colorbar(surface,shrink=0.75)
    
def plotLines(lines, x, title, xlabel, ylabel, legend):
    assert len(lines) == len(legend)
    pylab.figure(figsize = (12,6))
    for line in lines:
        pylab.plot(x, line)
    pylab.xlabel(xlabel, fontsize = 15)
    pylab.ylabel(ylabel, fontsize = 15)
    pylab.title(title, fontproperties = font)
    pylab.legend(legend, fontsize = 15)
