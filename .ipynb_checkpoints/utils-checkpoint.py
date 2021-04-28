import numpy as np 
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std() 

    #Plot rolling statistics:
    f,ax=plt.subplots(1,1,figsize=(20,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
def difference(dataset, interval=365):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset.iloc[i,0] - dataset.iloc[i - interval,0]
        diff.append(value)
    return pd.DataFrame(diff)

def distplot_probplot(df, target):    
    """
     Plot histogram using normal distribution and probability plot
    """
    
    fig, ax = plt.subplots(1,2, figsize= (15,5))
    fig.suptitle("SalesPrice Normal Distribution and Probability Plot", fontsize= 15)
    
    # Plot using normal distribution 
    sns.distplot(df[target], fit=stats.norm,label='test_label2', ax = ax[0])
      
    # Probabiloty plot
    stats.probplot(df[target], plot = ax[1])
    
    plt.show()  
    
    # Get the normal distribution fitted parameters
    (mu, sigma) = stats.norm.fit(df[target])
    print('mean= {:.2f}, sigma= {:.2f}, mode= {:.2f})'.format(mu, sigma, stats.mode(df[target])[0][0]))
    
def normality_stats(df, target):
    """
    Get Skewness, Kurtosis test stats.
    """

    print(f"Skewness: {abs(df[target]).skew()}")
    print(f"Kurtosis: {abs(df[target]).kurt()}")


def plot_character(df, feature, target, i):
    f, ax = plt.subplots(1,2, figsize = (14,4))
    sns.boxplot(data = df, x = feature , y= target, ax = ax[0])
    sns.violinplot(data = df, x = feature , y= target, ax = ax[1])
    # plt.show()
    f.savefig("plots/character/{}-{}.png".format(i,feature))
    plt.close()
    
def plot_numeric(df, feature, target, i):
    f, ax = plt.subplots(1,2, figsize = (14,4))
    f.tight_layout(pad=4.0) # To increase the space between subplots
    
    # 1) Histogram Plot
    sns.distplot(df[feature], kde=False, ax = ax[0])
    ax[0].set(xlabel=feature, title='Histogram')
    
    # 2) regression plot
    corr = round(df[[feature, target]].corr().iloc[0,1],3)
    sns.regplot(data=df, x=feature, y=target, scatter_kws={'alpha':0.2}, line_kws={'color': 'blue'}, ax = ax[1]) 
    ax[1].set_title('{} vs Rented bikes (Correlation coefficient: {})'.format(feature, corr), fontsize = 14)

    # plt.show()
    f.savefig("plots/numeric/{}-{}.png".format(i,feature))
    plt.close()
    
def rmsle_cv(model):
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train.values, scoring="neg_mean_squared_error", cv = kf))
    
    return(rmse)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculates MAPE given y_true and y_pred
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100