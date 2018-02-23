import numpy as np
import pandas as pd

start = '2011-01-01'                 #起始时间
end = '2016-01-01'                   #结束时间
benchmark = 'HS300'                  #策略参考标准
universe = set_universe('HS300')     #证券池，支持股票和基金
capital_base = 10000000              #起始资金
freq = 'd'                           #策略类型,'d'--日线回测,'m'--分钟线回测
refresh_rate = 20                    #调仓频率,时间间隔,freq = 'd'间隔为交易日，freq = 'm'间隔为分钟

def initialize(account):             # 初始化虚拟账户状态
    pass

def handle_data(account):            # 每个交易日的买入卖出指令
    
    # 获取上一个交易日
    yesterday = account.privious_date.striftime('%Y%m%d')
    
    # 市盈率
    PE=DataAPI.MktStockFactorsOneDayGet(stradeDate = yesterday, secID = account.universe, field = u'secID,PE', pandas='1').set_index('secID')
    PE = 1.0 / PE
    ep = PE['PE'].dropna().to_dict()
    signal_PE = pd.Series(standardize(neutralize(winsorize(ep), yesterday)))
    
    # 对数流通市值
    LFLO = DataAPI.MktStockFactorsOneDayGet(tradeDate=yesterday,secID=account.universe,field=u'secID,LFLO',pandas='1').set_index('secID')
    LFLO = 1.0 / LFOL
    lflo = LFLO['LFLO'].dropna().to_dict()
    signal_LFLO = pd.Series(standardize(winsorize(lflo)))
    
    total_score = (signal_PE + signal_LFLO)*0.5
    wts = simple_long_only(total_score.dropna().to_dict(), yesterday)
    
    # 先卖出
    buy_list = wts.keys()
    for stk in account.valid_secpos:
        if stk not in buy_list:
           order_to(stk, 0)
    
    # 再买入
    total_money = account.referencePortfolioValue
    prices = account.referencePrice
    for stk in buy_list:
        if np.isnan(prices[stk]) or prices[stk] == 0:  # 停牌或是还没有上市等原因不能交易
            continue
        order_to(stk, int(total_money * wts[stk] / prices[stk] / 100) * 100)

