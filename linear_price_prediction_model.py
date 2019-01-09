#!/usr/bin/env python
""" Simple linear stock price prediction model.

    Initial goal was to model using the Euler-Maruyama method to solve the 
    Ornstein-Uhlenbeck process and fit this to real data to extract drift and
    diffusion coefficients. This led to unpredictable results when tested due 
    to the stochastic nature of the process. 
    
    Using data from Quandl I instead look at the daily returns (close to close)
    and perform a linear regression on the (exponentially) weighted average of 
    the past N days (can be optimized as well).
    
    This predictived model is then be used to develop a trading strategy, not
    fully functioning at the moment.
    
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import quandl as qd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression

N = 1000    #Array length  
dt = 1./N   #Time steps
#seedVal = 2
#np.random.seed(seedVal)
randArr = np.random.normal(0,np.sqrt(dt),N)

mu = 0.2     #
sigma = 0.4

wn = np.zeros(N)    #Wiener process array    
xn = np.zeros(N)    #Price array
xn[0] = 1           #Initial price

    # Euler-maruyama method:
    # X[n+1] = X[n] + r*f(X[n])*dt + sigma*dW(n)*X[n]
    # f is the 'restoring force' function
    # g is the forcing function, in this case white Guassian noise
    # i.e. Brownian motion
    #
    # *** removed this whole section because all we care about are results ***
    # ************************************************************************
    
"""Now let's get some real data using quandl"""

startDate = "2017-01-01"
endDate = "2017-12-31"

df = qd.get("WIKI/F", start_date = startDate, end_date = endDate) 

time = np.linspace(1, len(df['Adj. Close']), len(df['Adj. Close']))
returns = pd.Series.diff(df['Adj. Close'])/df['Adj. Close']

# Shift so that we're predicting today's close price from yesterday's returns:
returns = returns.shift(-1)[:-1]

k = 5 # Rolling average length
        
sigma = pd.ewmstd(returns, span = k)*k  # 'diffusion' coefficient
mu = pd.ewma(returns, span = k)*k       # 'drift' coefficient

#del sigma.index.name

# New length:
N = len(df['Adj. Close'])
# Number of sims to run:
M = 100
XN = np.zeros([M,N])
# New time steps:
dt = 1./N


""" NOT USED """
#xn = np.zeros(N)
#xn[k] = df['Adj. Close'][k]
#
#for j in range(M):
#    for i in range(k+1,N):
#        dw = randArr[i]
#        xn[i] = xn[i-1] + mu[i]*dt*xn[i-1] + sigma[i]*np.random.normal(mu[i],np.sqrt(dt))*xn[i-1]
#    XN[j] = xn
#    
#plt.plot(time, df['Adj. Close'])
#plt.plot(time[k:],np.mean(XN,axis=0)[k:],'.')
#
#eSqr = np.sum((np.mean(XN,axis=0)[k:]-df['Adj. Close'][k:].values)**2)
#print eSqr

"Let's make a new dataframe with mu and sigma and the returns each day"
df2 = sigma[k:].to_frame(name='Sigma').join(mu[k:].to_frame(name='Mu')).join(returns[k:].to_frame(name='Returns'))

x = df2.drop('Returns',axis=1)
y = df2['Returns']

outCorr = np.zeros(M)

for j in range(M):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    
    # Try a linear regression model:
    lm = LinearRegression()
    
    lm.fit(x_train,y_train)
    lm_preds = lm.predict(x_test)
    
    predFrame = pd.Series(lm_preds).to_frame(name='Predictions')
    
    results = y_test.to_frame().join(predFrame)
    results.corr()
    
    dfCorr = results.corr()
    outCorr[j] = dfCorr['Predictions'][0]
    
# THIS IS WHAT WE CARE ABOUT (Pearson correlation coefficient):
np.mean(outCorr)

""" Now let's test it on another year and see how well it holds up:"""
startDateNew = "2018-01-01"
endDateNew = "2018-10-01"

dfNew = qd.get("WIKI/F", start_date = startDateNew, end_date = endDateNew)

timeNew = np.linspace(1, len(dfNew['Adj. Close']), len(dfNew['Adj. Close']))
returnsNew = pd.Series.diff(dfNew['Adj. Close'])/dfNew['Adj. Close']
returnsNew = returnsNew.shift(-1)[:-1]

k = 6 # Rolling average length
        
sigmaNew = pd.ewmstd(returnsNew, span = k)*k  # 'diffusion' coefficient
muNew = pd.ewma(returnsNew, span = k)*k       # 'drift' coefficient

# Let's make a new dataframe with mu and sigma and the returns each day
df2New = sigmaNew[k:].to_frame(name='Sigma').join(muNew[k:].to_frame(name='Mu')).join(returnsNew[k:].to_frame(name='Returns'))

xNew = df2New.drop('Returns',axis=1)
yNew = df2New['Returns']

lm_predsNew = lm.predict(xNew)

predFrameNew = pd.Series(lm_predsNew).to_frame(name='Predictions')
predFrameNew.index = yNew.index    

resultsNew = yNew.to_frame().join(predFrameNew)
resultsNew.corr()

# Let's plot the predicted vs actual daily returns:
plt.figure
plt.plot(resultsNew['Predictions'],resultsNew['Returns'],'.')
plt.axhline(0, linestyle='--', color='k')
plt.axvline(0, linestyle='--', color='k')
plt.xlabel('Predicted daily returns')
plt.ylabel('Actual daily returns')


# Now let's try a strategy trading Ford stock in 2018...

""" 
 Now let's iterate through each day and predict whether we should go short
 or long the following day and by how much. For that we'll just scale by the
 projected change for positive returns (going long) and 1/2 of the projected 
 change for negative returns. We'll scale this as a fraction of our total 
 available capital by using a maximum projected change of 7%, i.e. a projected
 change of 7% either way and we go all in.

 Let's assume we buy at the beginning of the day and sell at the end, although
 the actual holding time can be varied according to the data available.
 
 We also assume that our orders are immediately filled at the open/close price.
 
 This part isn't working well yet, have to come back to it.
"""

P = np.zeros(np.size(timeNew)-1)
P[k] = 1000  # Initial total capital available
P[k+1] = 1000  # Initial total capital available
yLivePred = np.zeros(np.size(timeNew)-1)
sharesBought = np.zeros(np.size(timeNew)-1)
sharesShorted = np.zeros(np.size(timeNew)-1)

for i in range(k+1,np.size(timeNew)-1):
    # Get the new sigma and mu for today (using the data up until yesterday)
    sigmaLive = pd.ewmstd(returnsNew[:i-1], span = k)*k
    muLive = pd.ewma(returnsNew[:i-1], span = k)*k
    xLive = sigmaLive.to_frame(name='Sigma').join(muLive.to_frame(name='Mu'))
    
    # Predict the next price...
    yLivePred[i] = lm.predict(xLive[i-2:i])
    
    if i > k+1:
        if sharesBought[i-1] > 0:
            
            # Sell everything if decrease predicted
            if yLivePred[i] < 0: 
                P[i] = P[i-1]
                # Sell previous shares at new price and add profits
                P[i] += sharesBought[i] * dfNew['Adj. Close'][i]
                sharesBought[i] = 0
                
                # Then short more shares:
                sharesShorted[i] = np.floor(P[i]*abs(yLivePred[i]/0.2)/dfNew['Adj. Close'][i])
                P[i] -= sharesShorted[i]*dfNew['Adj. Close'][i]

            # Instead buy more if another increase is predicted  
            elif yLivePred[i] > 0: 
                sharesBought[i] = sharesBought[i-1] + np.floor(P[i]*(yLivePred[i]/0.02)/dfNew['Adj. Close'][i])
                P[i] -= (sharesBought[i]-sharesBought[i-1])*dfNew['Adj. Close'][i]

        elif sharesShorted[i-1] > 0:            
            # Close position if increase predicted
            if yLivePred[i] > 0: 
                P[i] = P[i-1]
                
                P[i] += sharesShorted[i-1] * (-dfNew['Adj. Close'][i])
                sharesShorted[i] = 0
                
                # Then buy more shares:
                sharesBought[i] = np.floor(P[i]*(yLivePred[i]/0.02)/dfNew['Adj. Close'][i])
                P[i] -= sharesBought[i]*dfNew['Adj. Close'][i]
                
            # Instead short more if another decrease is predicted
            elif yLivePred[i] < 0:
                sharesShorted[i] = sharesShorted[i-1] + np.floor(P[i]*abs(yLivePred[i]/0.2)/dfNew['Adj. Close'][i])
                P[i] = P[i-1] - (sharesShorted[i]-sharesShorted[i-1])*dfNew['Adj. Close'][i]
        else:
            if yLivePred[i] >= 0:
                sharesBought[i] = sharesBought[i-1] + np.floor(P[i]*(yLivePred[i]/0.02)/dfNew['Adj. Close'][i])    
                P[i] = P[i-1] - (sharesBought[i]-sharesBought[i-1])*dfNew['Adj. Close'][i]
            elif yLivePred[i] < 0:
                sharesShorted[i] = sharesShorted[i-1] + np.floor(P[i]*abs(yLivePred[i]/0.2)/dfNew['Adj. Close'][i])
                P[i] = P[i-1] - (sharesShorted[i]-sharesShorted[i-1])*dfNew['Adj. Close'][i]
    else:
        if yLivePred[i] > 0:
            sharesBought[i] = sharesBought[i-1] + np.floor(P[i]*(yLivePred[i]/0.02)/dfNew['Adj. Close'][i])    
            P[i] = P[i-1] - (sharesBought[i]-sharesBought[i-1])*dfNew['Adj. Close'][i]
        elif yLivePred[i] < 0:
            sharesShorted[i] = sharesShorted[i-1] + np.floor(P[i]*abs(yLivePred[i]/0.2)/dfNew['Adj. Close'][i])
            P[i] = P[i-1] - (sharesShorted[i]-sharesShorted[i-1])*dfNew['Adj. Close'][i]
            
plt.plot(P[k+1:])


