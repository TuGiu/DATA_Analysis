import pandas as pd
import numpy as np
from pandas import Series, DataFrame
#d = {'one': Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': Series([1., 2., 3., 4.], index=['a','b', 'c', 'd'])}
#df = DataFrame(d, index=['r', 'd', 'a'], columns=['two', 'three'])
#df = DataFrame(d)
#print df.index
#print df.columns
#print df.values

#d = {'one': [1., 2., 3., 4.], 'two': [4., 3., 2., 1.]}
# df = DataFrame(d, index=['a', 'b', 'c', 'd'])
# print df

#df = DataFrame()
#print df

# a = Series(range(5))
# b = Series(np.linspace(4, 20, 5))
# df = pd.concat([a, b], axis=1)
# print df

df = DataFrame()
index = ['alpha', 'beta', 'gamma', 'delta', 'eta']
for i in range(5):
    a = DataFrame([np.linspace(i, 5*i, 5)], index=[index[i]])
    df = pd.concat([df, a], axis=0)
print df 
# print df[1]
df.columns = ['a', 'b', 'c', 'd', 'e']
# print df['b']
# df.b
# print type(df.b)
# df[['a', 'd']]
# print type(df[['a', 'd']])

# print df['b'][2]
# print df['a']['gamma']

#print df.iloc[1]
# print df.loc['beta']

# print df[1:3]
# bool_vec = [True, False, True, True, False]
# print df[bool_vec]

# print df[['b', 'd']].iloc[[1, 3]]
# print df.iloc[[1, 3]][['b', 'd']]
# print df[['b', 'd']].loc[['beta', 'delta']]
# print df.loc[['beta', 'delta']][['b', 'd']]

# print df.iat[2,3]
# print df.at['gamma', 'd']

# print df.ix['gamma', 4]
# print df.ix[['delta', 'gamma'], [1, 4]]
# print df.ix[[1, 2], ['b', 'e']]
# print df.ix[['beta', 2], ['b', 'e']]
# print df.ix[[1,2], ['b', 4]]