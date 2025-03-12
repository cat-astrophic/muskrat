# This script performs the time series analyses on the cleaned twitter data

# Importing required modules

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as stats
from matplotlib import pyplot as plt
from pylab import rcParams

# Project directory

direc = 'D:/muskrat/'

# Get TSLA and other stock data

tix = ['TSLA', 'TM', 'VWAGY', 'GM', 'HMC', 'F', 'SZKMY', 'FUJHY', 'RIVN', 'ISUZY', 'YMHAY']

stock = yf.download(tix, start = '2022-10-01', end = '2022-11-30')

# Reduce stock to only contain closing prices

stock = stock[stock.columns[:int((np.shape(stock)[1]-1)/5)+1]]

# Creating a dataframe of stock data

days = stock.index.to_list()
days = [str(d)[:10] for d in days]
df = pd.DataFrame(pd.Series(days, name = 'Date'))

for i in range(np.shape(stock)[1]): 
    
    fuck = stock[stock.columns[i]]
    df = pd.concat([df, pd.Series(fuck.values.tolist(), name = sorted(tix)[i])], axis = 1)

# Reading in the sentiment analysis results

data = pd.read_csv('C:/Users/Michael/Documents/Data/muskrat/data/sentiment_data.csv')

# Create dates without time

dates = [d[:10] for d in data.Datetime]

data = pd.concat([data, pd.Series(dates, name = 'Date')], axis = 1)

# Create daily data by score type

nega = []
neut = []
posi = []

ang = []
dis = []
neg = []
joy = []
pos = []
fea = []
sad = []
tru = []
sur = []

for i in range(len(df.Date)):
    
    print(df.Date[i])
    tmp = data[data.Date <= df.Date[i]]
    
    if i > 0:
        
        tmp = tmp[tmp.Date > df.Date[i-1]]
        
    nega.append(tmp.Negative.sum())
    neut.append(tmp.Neutral.sum())
    posi.append(tmp.Positive.sum())
    ang.append(tmp.Anger.sum())
    dis.append(tmp.Disgust.sum())
    neg.append(tmp.Negative_E.sum())
    joy.append(tmp.Joy.sum())
    pos.append(tmp.Positive_E.sum())
    fea.append(tmp.Fear.sum())
    sad.append(tmp.Sadness.sum())
    tru.append(tmp.Trust.sum())
    sur.append(tmp.Surprise.sum())

nega = pd.Series(nega, name = 'Negative')
neut = pd.Series(neut, name = 'Neutral')
posi = pd.Series(posi, name = 'Positive')
ang = pd.Series(ang, name = 'Anger')
dis = pd.Series(dis, name = 'Disgust')
neg = pd.Series(neg, name = 'Negative_E')
joy = pd.Series(joy, name = 'Joy')
pos = pd.Series(pos, name = 'Positive_E')
fea = pd.Series(fea, name = 'Fear')
sad = pd.Series(sad, name = 'Sadness')
tru = pd.Series(tru, name = 'Trust')
sur = pd.Series(sur, name = 'Surprise')

df = pd.concat([df, nega, neut, posi, ang, dis, neg, joy, pos, fea, sad, tru, sur], axis = 1)

# Creating market trend data

trend = []

for i in range(len(df)):
    
    x = df.iloc[i]
    trend.append(x[1:12].mean())

df = pd.concat([df, pd.Series(trend, name = 'MT')], axis = 1)

# Creating change in market trend data

ct = [None]

for i in range(1,len(df)):
    
    ct.append(df.MT[i] - df.MT[i-1])

df = pd.concat([df, pd.Series(ct, name = 'DMT')], axis = 1)

# Creating lag price data

lags = [None]

for i in range(1,len(df)):
    
    lags.append(df.TSLA[i-1])

df = pd.concat([df, pd.Series(lags, name = 'Lag')], axis = 1)

# Adding a fixed effect for pre/post buyout

df = pd.concat([df, pd.Series([0]*19 + [1]*22, name = 'Post')], axis = 1)

# Normalizing sentiment scores

df.Positive = df.Positive / 1000
df.Negative = df.Negative / 1000
df.Neutral = df.Neutral / 1000

df.Positive_E = df.Positive_E / 1000
df.Negative_E = df.Negative_E / 1000
df.Anger = df.Anger / 1000
df.Disgust = df.Disgust / 1000
df.Joy = df.Joy / 1000
df.Fear = df.Fear / 1000
df.Sadness = df.Sadness / 1000
df.Trust = df.Trust / 1000
df.Surprise = df.Surprise / 1000

# Baseline regressions

reg = df.iloc[1:]

Y = reg['TSLA']
X0 = stats.add_constant(reg[['Lag', 'DMT']])
X1 = stats.add_constant(reg[['Lag', 'DMT', 'Post']])
X2 = stats.add_constant(reg[['Lag', 'DMT', 'Post', 'Positive', 'Negative', 'Neutral']])
X3 = stats.add_constant(reg[['Lag', 'DMT', 'Post', 'Anger', 'Disgust', 'Negative_E', 'Joy', 'Positive_E', 'Fear', 'Sadness', 'Trust', 'Surprise']])

res0 = stats.OLS(Y,X0).fit(cov_type = 'HC1')
res1 = stats.OLS(Y,X1).fit(cov_type = 'HC1')
res2 = stats.OLS(Y,X2).fit(cov_type = 'HC1')
res3 = stats.OLS(Y,X3).fit(cov_type = 'HC1')

print(res0.summary())
print(res1.summary())
print(res2.summary())
print(res3.summary())

# Elasticity regressions

this_fucker = []

for i in range(1,len(df)):
    
    this_fucker.append(np.log(df.MT[i]) - np.log(df.MT[i-1]))

reg = reg.reset_index(drop = True)

reg = pd.concat([reg, pd.Series(np.log(reg.TSLA), name = 'lnTSLA'),
                 pd.Series(np.log(reg.Lag), name = 'lnLag'),
                 pd.Series(this_fucker, name = 'lnDMT'),
                 pd.Series(np.log(reg.Positive), name = 'lnPositive'),
                 pd.Series(np.log(reg.Negative), name = 'lnNegative'),
                 pd.Series(np.log(reg.Neutral), name = 'lnNeutral'),        
                 pd.Series(np.log(reg.Anger), name = 'lnAnger'),
                 pd.Series(np.log(reg.Disgust), name = 'lnDisgust'),
                 pd.Series(np.log(reg.Negative_E), name = 'lnNegative_E'),
                 pd.Series(np.log(reg.Joy), name = 'lnJoy'),
                 pd.Series(np.log(reg.Positive_E), name = 'lnPositive_E'),
                 pd.Series(np.log(reg.Fear), name = 'lnFear'),
                 pd.Series(np.log(reg.Sadness), name = 'lnSadness'),
                 pd.Series(np.log(reg.Trust), name = 'lnTrust'),
                 pd.Series(np.log(reg.Surprise), name = 'lnSurprise')], axis = 1)

YY = reg['lnTSLA']
XX0 = stats.add_constant(reg[['lnLag', 'lnDMT']])
XX1 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post']])
XX2 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post', 'lnPositive', 'lnNegative', 'lnNeutral']])
XX3 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post', 'lnAnger', 'lnDisgust', 'lnNegative_E', 'lnJoy', 'lnPositive_E', 'lnFear', 'lnSadness', 'lnTrust', 'lnSurprise']])

rres0 = stats.OLS(YY,XX0).fit(cov_type = 'HC1')
rres1 = stats.OLS(YY,XX1).fit(cov_type = 'HC1')
rres2 = stats.OLS(YY,XX2).fit(cov_type = 'HC1')
rres3 = stats.OLS(YY,XX3).fit(cov_type = 'HC1')

print(rres0.summary())
print(rres1.summary())
print(rres2.summary())
print(rres3.summary())

# Saving results

res_list = [res0, res1, res2, res3]
rres_list = [rres0, rres1, rres2, rres3]

for r in res_list:
    
    file = open(direc + 'results/res' + str(res_list.index(r)) + '.txt', 'w')
    file.write(r.summary().as_text())
    file.close()

for r in rres_list:
    
    file = open(direc + 'results/rres' + str(rres_list.index(r)) + '.txt', 'w')
    file.write(r.summary().as_text())
    file.close()

# Creating a time series plot for each of the three time series with two scales

rcParams['figure.figsize'] = 8.5, 8.5
cm = plt.get_cmap('gist_rainbow')

plt.figure(0)
fig, ax1 = plt.subplots()
basis = [i for i in range(41)]
plt.plot(basis, df.TSLA, label = 'TSLA Stock Price', color = 'k')
plt.ylabel('USD')
plt.xlabel('Date')
plt.vlines(x = 19, ymin = 0, ymax = 280, color = 'k')
tix = [0, 5, 10, 15, 20, 25, 30, 35, 39]
labels = ['Oct 3', ' Oct 10', 'Oct 17', 'Oct 24', 'Oct 31', 'Nov 7', 'Nov 14', 'Nov 21', 'Nov 28']
plt.xticks(tix, labels)
ax2 = ax1.twinx()
plt.plot(basis, 1000*df.Positive, label = 'Positive Sentiment', color = cm(180))
plt.plot(basis, 1000*df.Negative, label = 'Negative Sentiment', color = cm(0))
plt.ylabel('Aggregate Sentiment')
plt.title('A Comparison of the price of TSLA and public perception of Elon Musk', loc = 'center', fontsize = 12, fontweight = 40, color = 'black')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.vlines(x = 19, ymin = 0, ymax = 300000, color = 'k')
ax2.legend(lines1 + lines2, labels1 + labels2)
fig.tight_layout()
plt.savefig(direc + '/figures/time_series_plot.jpg')

