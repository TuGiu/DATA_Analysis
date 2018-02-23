#回测--未完成（出错）

import numpy as np
import pandas as pd
import datetime as dt
from pandas import Series, DataFrame, isnull
from datetime import timedelta, datetime
from CAL.PyCAL import *

pd.set_option('display.width', 200)

universe = set_universe('HS300')

today = Date.todaysDate()
start_date = (today - Period('1Y')).toDateTime().strftime('%Y%m')
#print start_date
end_date = today.toDateTime().strftime('%Y%m%d')
#print end_date

market_capital = DataAPI.MktEqudGet(secID=universe, field=['secID', 'tradeDate', 'marketValue', 'negMarketValue'], beginDate=start_date, endDate=end_date, pandas='1')

equity = DataFrame()
for rpt_type in ['Q1', 'S1', 'Q3', 'A']:
    try:
        tmp = DataAPI.FdmtBSGet(secID=universe, field=['secID', 'endDate', 'publishDate', 'TEquityAttrP'], beginDate=start_date, publishDateEnd=end_date, reportType=rpt_type)
    except:
        tmp = DataFrame()
    equity = pd.concat([equity, tmp], axis=0)

print 'Data of TEquityAttrP:'
print equity.head()
print 'Date of marketValue:'
print market_capital.head()