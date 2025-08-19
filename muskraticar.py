# This script performs the time series analyses on the cleaned twitter data

# Importing required modules

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as stats
from statsmodels.tsa.stattools import grangercausalitytests as gc
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

data = pd.read_csv(direc + 'data/sentiment_data.csv')

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

# Creating lag sentiment data

lag_nega = [None]
lag_neut = [None]
lag_posi = [None]

lag_ang = [None]
lag_dis = [None]
lag_neg = [None]
lag_joy = [None]
lag_pos = [None]
lag_fea = [None]
lag_sad = [None]
lag_tru = [None]
lag_sur = [None]

for i in range(1,len(df)):
    
    lag_nega.append(df.Negative[i-1])
    lag_neut.append(df.Neutral[i-1])
    lag_posi.append(df.Positive[i-1])
    
    lag_ang.append(df.Anger[i-1])
    lag_dis.append(df.Disgust[i-1])
    lag_neg.append(df.Negative_E[i-1])
    lag_joy.append(df.Joy[i-1])
    lag_pos.append(df.Positive_E[i-1])
    lag_fea.append(df.Fear[i-1])
    lag_sad.append(df.Sadness[i-1])
    lag_tru.append(df.Trust[i-1])
    lag_sur.append(df.Surprise[i-1])

df = pd.concat([df, pd.Series(lag_nega, name = 'Negative_Lag'), pd.Series(lag_neut, name = 'Neutral_Lag'),
                pd.Series(lag_pos, name = 'Positive_Lag'), pd.Series(lag_ang, name = 'Anger_Lag'),
                pd.Series(lag_dis, name = 'Disgust_Lag'), pd.Series(lag_neg, name = 'Negative_E_Lag'),
                pd.Series(lag_joy, name = 'Joy_Lag'), pd.Series(lag_pos, name = 'Positive_E_Lag'),
                pd.Series(lag_fea, name = 'Fear_Lag'), pd.Series(lag_sad, name = 'Sadness_Lag'),
                pd.Series(lag_tru, name = 'Trust_Lag'), pd.Series(lag_sur, name = 'Surprise_Lag')], axis = 1)

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

# Adding a variable for the earnings call

earn = [1 if i == 12 else 0 for i in range(len(df))]
ep = [1 if i >= 12 else 0 for i in range(len(df))]

df = pd.concat([df, pd.Series(earn, name = 'EC'), pd.Series(ep, name = 'EC_Post')], axis = 1)

# Baseline regressions

reg = df.iloc[1:]

Y = reg['TSLA']
X0 = stats.add_constant(reg[['Lag', 'DMT']])
X1 = stats.add_constant(reg[['Lag', 'DMT', 'Post', 'EC']])
X2 = stats.add_constant(reg[['Lag', 'DMT', 'Post', 'EC', 'Positive', 'Negative', 'Neutral']])
X3 = stats.add_constant(reg[['Lag', 'DMT', 'Post', 'EC', 'Positive', 'Negative', 'Neutral', 'Positive_Lag', 'Negative_Lag', 'Neutral_Lag']])
X4 = stats.add_constant(reg[['Lag', 'DMT', 'Post', 'EC', 'Anger', 'Disgust', 'Negative_E', 'Joy', 'Positive_E', 'Fear', 'Sadness', 'Trust', 'Surprise']])
X5 = stats.add_constant(reg[['Lag', 'DMT', 'Post', 'EC', 'Anger', 'Disgust', 'Negative_E', 'Joy', 'Positive_E', 'Fear', 'Sadness', 'Trust', 'Surprise', 'Anger_Lag', 'Disgust_Lag', 'Negative_E_Lag', 'Joy_Lag', 'Positive_E_Lag', 'Fear_Lag', 'Sadness_Lag', 'Trust_Lag', 'Surprise_Lag']])

res0 = stats.OLS(Y,X0).fit(cov_type = 'HC1')
res1 = stats.OLS(Y,X1).fit(cov_type = 'HC1')
res2 = stats.OLS(Y,X2).fit(cov_type = 'HC1')
res3 = stats.OLS(Y,X3).fit(cov_type = 'HC1')
res4 = stats.OLS(Y,X4).fit(cov_type = 'HC1')
res5 = stats.OLS(Y,X5).fit(cov_type = 'HC1')

print(res0.summary())
print(res1.summary())
print(res2.summary())
print(res3.summary())
print(res4.summary())
print(res5.summary())

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

reg = pd.concat([reg, pd.Series(np.log(reg.Positive_Lag), name = 'lnPositive_Lag'),
                 pd.Series(np.log(reg.Negative_Lag), name = 'lnNegative_Lag'),
                 pd.Series(np.log(reg.Neutral_Lag), name = 'lnNeutral_Lag'),        
                 pd.Series(np.log(reg.Anger_Lag), name = 'lnAnger_Lag'),
                 pd.Series(np.log(reg.Disgust_Lag), name = 'lnDisgust_Lag'),
                 pd.Series(np.log(reg.Negative_E_Lag), name = 'lnNegative_E_Lag'),
                 pd.Series(np.log(reg.Joy_Lag), name = 'lnJoy_Lag'),
                 pd.Series(np.log(reg.Positive_E_Lag), name = 'lnPositive_E_Lag'),
                 pd.Series(np.log(reg.Fear_Lag), name = 'lnFear_Lag'),
                 pd.Series(np.log(reg.Sadness_Lag), name = 'lnSadness_Lag'),
                 pd.Series(np.log(reg.Trust_Lag), name = 'lnTrust_Lag'),
                 pd.Series(np.log(reg.Surprise_Lag), name = 'lnSurprise_Lag')], axis = 1)

YY = reg['lnTSLA']
XX0 = stats.add_constant(reg[['lnLag', 'lnDMT']])
XX1 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post', 'EC']])
XX2 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post', 'EC', 'lnPositive', 'lnNegative', 'lnNeutral']])
XX3 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post', 'EC', 'lnPositive', 'lnNegative', 'lnNeutral', 'lnPositive_Lag', 'lnNegative_Lag', 'lnNeutral_Lag']])
XX4 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post', 'EC', 'lnAnger', 'lnDisgust', 'lnNegative_E', 'lnJoy', 'lnPositive_E', 'lnFear', 'lnSadness', 'lnTrust', 'lnSurprise']])
XX5 = stats.add_constant(reg[['lnLag', 'lnDMT', 'Post', 'EC', 'lnAnger', 'lnDisgust', 'lnNegative_E', 'lnJoy', 'lnPositive_E', 'lnFear', 'lnSadness', 'lnTrust', 'lnSurprise', 'lnAnger_Lag', 'lnDisgust_Lag', 'lnNegative_E_Lag', 'lnJoy_Lag', 'lnPositive_E_Lag', 'lnFear_Lag', 'lnSadness_Lag', 'lnTrust_Lag', 'lnSurprise_Lag']])

rres0 = stats.OLS(YY,XX0).fit(cov_type = 'HC1')
rres1 = stats.OLS(YY,XX1).fit(cov_type = 'HC1')
rres2 = stats.OLS(YY,XX2).fit(cov_type = 'HC1')
rres3 = stats.OLS(YY,XX3).fit(cov_type = 'HC1')
rres4 = stats.OLS(YY,XX4).fit(cov_type = 'HC1')
rres5 = stats.OLS(YY,XX5).fit(cov_type = 'HC1')

print(rres0.summary())
print(rres1.summary())
print(rres2.summary())
print(rres3.summary())
print(rres4.summary())
print(rres5.summary())

# Log returns regressions

log_ret = [np.log(df.TSLA[i]) - np.log(df.TSLA[i-1]) for i in range(1,len(df))]
log_ret_lag = [None] + log_ret[:len(log_ret)-1]

reg = pd.concat([reg, pd.Series(log_ret, name = 'Log_Return'),
                 pd.Series(log_ret_lag, name = 'Log_Return_Lag')], axis = 1)

reg = reg[1:]

YYY = reg['Log_Return']
XXX0 = stats.add_constant(reg[['Log_Return_Lag', 'lnDMT']])
XXX1 = stats.add_constant(reg[['Log_Return_Lag', 'lnDMT', 'Post', 'EC']])
XXX2 = stats.add_constant(reg[['Log_Return_Lag', 'lnDMT', 'Post', 'EC', 'lnPositive', 'lnNegative', 'lnNeutral']])
XXX3 = stats.add_constant(reg[['Log_Return_Lag', 'lnDMT', 'Post', 'EC', 'lnPositive', 'lnNegative', 'lnNeutral', 'lnPositive_Lag', 'lnNegative_Lag', 'lnNeutral_Lag']])
XXX4 = stats.add_constant(reg[['Log_Return_Lag', 'lnDMT', 'Post', 'EC', 'lnAnger', 'lnDisgust', 'lnNegative_E', 'lnJoy', 'lnPositive_E', 'lnFear', 'lnSadness', 'lnTrust', 'lnSurprise']])
XXX5 = stats.add_constant(reg[['Log_Return_Lag', 'lnDMT', 'Post', 'EC', 'lnAnger', 'lnDisgust', 'lnNegative_E', 'lnJoy', 'lnPositive_E', 'lnFear', 'lnSadness', 'lnTrust', 'lnSurprise', 'lnAnger_Lag', 'lnDisgust_Lag', 'lnNegative_E_Lag', 'lnJoy_Lag', 'lnPositive_E_Lag', 'lnFear_Lag', 'lnSadness_Lag', 'lnTrust_Lag', 'lnSurprise_Lag']])

rrres0 = stats.OLS(YYY,XXX0).fit(cov_type = 'HC1')
rrres1 = stats.OLS(YYY,XXX1).fit(cov_type = 'HC1')
rrres2 = stats.OLS(YYY,XXX2).fit(cov_type = 'HC1')
rrres3 = stats.OLS(YYY,XXX3).fit(cov_type = 'HC1')
rrres4 = stats.OLS(YYY,XXX4).fit(cov_type = 'HC1')
rrres5 = stats.OLS(YYY,XXX5).fit(cov_type = 'HC1')

print(rrres0.summary())
print(rrres1.summary())
print(rrres2.summary())
print(rrres3.summary())
print(rrres4.summary())
print(rrres5.summary())

# Saving results

res_list = [res0, res1, res2, res3, res4, res5]
rres_list = [rres0, rres1, rres2, rres3, rres4, rres5]
rrres_list = [rrres0, rrres1, rrres2, rrres3, rrres4, rrres5]

for r in res_list:
    
    file = open(direc + 'results/res' + str(res_list.index(r)) + '.txt', 'w')
    file.write(r.summary().as_text())
    file.close()

for r in rres_list:
    
    file = open(direc + 'results/rres' + str(rres_list.index(r)) + '.txt', 'w')
    file.write(r.summary().as_text())
    file.close()

for r in rrres_list:
    
    file = open(direc + 'results/rrres' + str(rrres_list.index(r)) + '.txt', 'w')
    file.write(r.summary().as_text())
    file.close()

# Granger causality testing

#gc1 = gc(reg[['lnTSLA', 'Trust']].dropna(), 8)
#gc2 = gc(reg[['lnTSLA', 'lnTrust']].dropna(), 8)
#gc3 = gc(reg[['Log_Return', 'Trust']].dropna(), 8)
#gc4 = gc(reg[['Log_Return', 'lnTrust']].dropna(), 8)

#gc5 = gc(reg[['Trust', 'lnTSLA']].dropna(), 8)
#gc6 = gc(reg[['lnTrust', 'lnTSLA']].dropna(), 8)
#gc7 = gc(reg[['Trust', 'Log_Return']].dropna(), 8)
#gc8 = gc(reg[['lnTrust', 'Log_Return']].dropna(), 8)

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

