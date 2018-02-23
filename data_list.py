import numpy as np
import pandas as pd
from pandas import Series, DataFrame


pd.set_option('display.width', 200)

#dates = pd.date_range('20150101', periods=5)
#print dates
#df = pd.DataFrame(np.random.randn(5,4), index=dates,columns=list('ABCD'))
# print df
#df2 = pd.DataFrame({ 'A' : 1., 'B': pd.Timestamp('20150214'), 'C':pd.Series(1.6,index=list(range(4)),dtype='float64'), 'D' : np.array([4] * 4,dtype='int64'), 'E':'hello pandas!'})
# print df2

stock_list = ['000001.XSHE', '0000002.XSHE', '000568.XSHE', '000625', '000768.XSHE', '600028.XSHG', '601111.XSHG', '601390.XSHG', '601998.XSHG']
raw_data = DataAPI.MktEqudGet(secID=stock_list, beginDate='20160101', endDate='20171231', pandas='1')
df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]

print df.shape
# print df
# print df.head(10)
# print df.tail(3)
# print df.describe()
# print df.sort_index(axis=1, ascending=False).head()

# print df.sort(columns='tradeDate').head()
# df = df.sort(columns=['tradeDate', 'secID'], ascending=[False, True])
# print df.head()

# print df.iloc[1:4][:]

# print df[df.closePrice > df.closePrice.mean()].head()
# print df[df.closePrice < df.closePrice.mean()].head()

# print df[df['secID'].isin(['601628.XSHG', '000001.XSHE', '600030.XSHG'])].head()
# print df.shape

# df['openPrice'][df['secID'] == '000001.XSHE'] = np.nan
# df['highestPrice'][df['secID'] == '601111.XSHG'] = np.nan
# df['lowestPrice'][df['secID'] == '601111.XSHG'] = np.nan 
# df['closePrice'][df['secID']  == '000002.XSHE'] = np.nan 
# df['turnoverVol'][df['secID'] == '601111.XSHG'] = np.nan 
#rint df.head(10)

# print df.dropna().shape
# print df.dropna().head(10)

# print df.dropna(how='all').shape
# print df.dropna(how='all').head(10)

# print df.dropna(thresh=6).shape
# print df.dropna(thresh=6).head

# print df.dropna(subset=['closePrice']).shape
# print df.dropna(subset=['closePrice']).head(10)

# print df.fillna(value=20160101).head()

df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]
# print df.mean(0)

# print df['closePrice'].value_counts().head()

#print df[['closePrice']].apply(lambda x: (x - x.min()) / (x.max() - x.min())).head()

#dat1 = df[['secID', 'tradeDate', 'closePrice']].head()
# dat2 = df[['secID', 'tradeDate', 'closePrice']].iloc[2]
# dat = dat1.append(dat2, ignore_index=True)
# print dat

# dat1 = df[['secID', 'tradeDate', 'closePrice']]
# dat2 = df[['secID', 'tradeDate', 'turnoverVol']]
# dat = dat1.merge(dat2, on=['secID', 'tradeDate'])
# print dat1.head()
# print dat2.head()
# print dat.head()

# df_grp = df.groupby('secID')
# grp_mean = df_grp.mean()
# print grp_mean

# df2 = df.sort(columns=['secID', 'tradeDate'], ascending=[True, False])
# print df2.drop_duplicates(subset='secID', take_last=True)

dat = df[df['secID'] == '600028.XSHG'].set_index('tradeDate')['closePrice']
dat.plot(title="Close Price of SINOPEC (600028) during Jan, 2017")
