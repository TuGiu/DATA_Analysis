import scipy
scipy.random.randn(10)

#pylab.figure(figsize = (12,3))
#randomSeries = scipy.random.randn(1000)
#pylab.plot(randomSeries)
#print u'均 值：%.4f' % randomSeries.mean()
#print u'标准差:%.4f' % randomSeries.std()

def call_option_pricer_monte_carlo(spot, strike, maturity, r, vol, numOfPath = 5000):
    randomSeries = scipy.random.randn(numOfPath)
    s_t = spot * np.exp((r - 0.5 * vol * vol) * maturity + randomSeries * vol * sqrt(maturity))
    sumValue = np.maximum(s_t - strike, 0.0).sum()
    price = exp(-r*maturity) * sumValue / numOfPath
    return price

#print '期权价格（蒙特卡洛）: %.4f' % call_option_pricer_monte_carlo(spot, strike, maturity, r, vol)

# 我们这里实验从1000次模拟到50000次模拟的结果，每次同样次数的模拟运行100遍
pathScenario = range(1000, 5000, 1000)
numberOfTrials = 100

confidenceIntervalUpper = []
confidenceIntervalLower = []
means = []   #means收入

for scenario in pathScenario:
    res = np.zeros(numberOfTrials)
    for i in range(numberOfTrials):
        res[i] = call_option_pricer_monte_carlo(spot, strike, maturity, r, vol, numOfPath = scenario)
    means.append(res.mean())
    confidenceIntervalUpper.append(res.mean() + 1.96*res.std())
    confidenceIntervalLower.append(res.mean() - 1.96*res.std())
    
#pylab.figure(figsize = (12,3))
#tabel = np.array([means,confidenceIntervalUpper, confidenceIntervalLower]).T
#pylab.plot(pathScenario, tabel)
#pylab.title(u'期权计算蒙特卡洛模拟', fontproperties = font, fontsize = 18)
#pylab.legend([u'均值', u'95%置信区间上界', u'95%置信区间下界'], prop = font)
#pylab.ylabel(u'价格', fontproperties = font, fontsize = 15)
#pylab.xlabel(u'模拟次数', fontproperties = font, fontsize = 15)
#pylab.grid(True)

from scipy.optimize import brentq

# 目标函数，目标价格由target确定
class cost_function:
    def __init__(self, target):
        self.targetValue = target
        
    def __call__(self, x):
        return call_option_pricer(spot, strike, maturity, r, x) - self.targetValue
    
# 假设我们使用vol初值作为目标
target = call_option_pricer(spot, strike, maturity, r, vol)
cost_sampel = cost_function(target)

# 使用Brent算法求解
impliedVol = brentq(cost_sampel, 0.01, 0.5)

print u'真实波动率： %.2f' % (vol*100,) + '%'
print u'隐含波动率： %.2f' % (impliedVol*100,) + '%'