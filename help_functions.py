import pandas as pd
import pandas.tseries
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns
import statsmodels.api as sm

# Dickey Fuller Test
from statsmodels.tsa.stattools import adfuller
# Auto Correlation
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# AR Autoregressive
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings

import pmdarima as pm
from pmdarima import model_selection
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima.arima.stationarity import ADFTest
from pmdarima.model_selection import train_test_split

from help_functions import *

warnings.filterwarnings('ignore')

def get_datetimes(df):
    """
    Takes a dataframe:
    returns only those column names that can be converted into datetime objects 
    as datetime objects.
    NOTE number of returned columns may not match total number of columns in passed dataframe
    """
    
    return pd.to_datetime(df.columns.values[1:], format='%Y-%m')

def melt_data(df):
    """
    Takes the zillow_data dataset in wide form or a subset of the zillow_dataset.  
    Returns a long-form datetime dataframe 
    with the datetime column names as the index and the values as the 'values' column.
    
    If more than one row is passes in the wide-form dataset, the values column
    will be the mean of the values from the datetime columns in all of the rows.
    """
    
    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 
                                  'State', 'Metro', 'CountyName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted #.groupby('time').aggregate({'value':'mean'})

def melt_data_roi(df):
    """
    Takes the zillow_data dataset in wide form or a subset of the zillow_dataset.  
    Returns a long-form datetime dataframe 
    with the datetime column names as the index and the values as the 'values' column.
    
    If more than one row is passes in the wide-form dataset, the values column
    will be the mean of the values from the datetime columns in all of the rows. For the top ROI
    """
    
    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 
                                  'State', 'Metro', 'CountyName', 'ROI'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted #.groupby('time').aggregate({'value':'mean'})

# plots Zip Code trends
def plot_trend(data, title):
    data.plot.line(color = 'green')
    plt.title(title)
    plt.ylabel('Price ($ in Thousands)')
    plt.grid()
    plt.show()

# plots mean of zip codes    
def plot_mean(df, col ,value):
    df.sort_values().plot.barh(color = 'g')
    plt.title(f'Average Value by {col}')
    plt.ylabel(f'{col}')
    plt.xlabel(f'Value ($) in {value}')
    return plt.show()

# sorts the value mean of each borough
def get_borough_mean(col):
    x = group[col].sort_values(ascending=False).head(10).round()
    return x

# ROI of each borough
def get_borough_ROI(df):
    x = df.sort_values(ascending=False).head(10)
    return x

#plots ROI
def plot_ROI(df, col):
    df.sort_values().plot.barh(color = 'g')
    plt.title(f'Average ROI Value by {col}')
    plt.ylabel(f'{col}')
    plt.xlabel(f'ROI %')
    return plt.show()

# Provided by canvas module
def stationarity_check(TS):
    
    # Import adfuller
    from statsmodels.tsa.stattools import adfuller
    
    # Calculate difference
    difference = TS.diff().dropna()
    
    # Perform the Dickey Fuller Test
    dftest = adfuller(TS.diff().dropna())
    
    # Plot Difference:
    fig = plt.figure(figsize=(12,6))
    plt.plot(difference, color='red', label='Difference in Lag')
    plt.legend(loc='best')
    plt.title('Difference in Lag')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print('Results of Dickey-Fuller Test: \n')

    dfoutput = pd.DataFrame(dftest[0:4], index=['Test Statistic',
                                            'p-value','#Lags Used','Number of Observations Used'], columns = ['Results'])
    print(dfoutput)
    
    return None

def model_validation(model, df, zipcode):
    """ visuallizes each model's validation """
    
    train = df[zipcode][:'2016-01-01']
    test = df[zipcode]['2016-01-01':]
    
    display(model.summary())
    predict = model.predict(start = pd.to_datetime('2010-01-01'), end = pd.to_datetime('2018-04-01'))
    
    plt.plot(predict[1:], label = 'Prediction')
    plt.plot(train, label = 'Train')
    plt.plot(test, label = 'Test')
    plt.title(f'Predictions for {zipcode}')
    plt.xlabel('Year')
    plt.ylabel('Price in $')
    plt.legend()
    return

def get_forecast(model, n, df, zipcode):
    """Gets the out of sample forecast for each model"""
    forecast = model.get_forecast(df, n)
    future_prediction = forecast.conf_int()
    future_prediction['value'] = forecast.predicted_mean
    future_prediction.columns = ['lower','upper','prediction'] 

    # Plotting our Forecast
    fig, ax = plt.subplots(figsize=(15, 7))
    df.plot(ax=ax,label='Real Values')

    future_prediction['prediction'].plot(ax=ax,label='predicted value',ls='--')

    ax.fill_between(x= future_prediction.index, y1= future_prediction['lower'], 
                y2= future_prediction['upper'],color='k',
                label='Confidence Interval')
    ax.legend() 
    plt.ylabel("Average Price")
    plt.title(f'Average Home Price - {zipcode} - With Forcasted Value & Confidence Intervals')
    plt.grid()
    plt.show()
    return 
    
# returns the in sample and out of sample predictions of the model    
def prediction_test(model, df, zipcode):
    train = df[zipcode][:'2016-01-01']
    test = df[zipcode]['2016-01-01':]
    
    pred = model.get_prediction(start=test.index.min(), 
          end=test.index.max(), dynamic = False)
    pred_conf = pred.conf_int()

    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot observed values
    ax = df[zipcode].plot(label='observed')

    # Plot predicted values
    pred.predicted_mean.plot(ax=ax, label='Prediction Series', alpha=0.9)

    # Plot the range for confidence intervals
    ax.fill_between(pred_conf.index,
                pred_conf.iloc[:, 0],
                pred_conf.iloc[:, 1], color='k', alpha=0.5,label = 'Confidence Interval')

    # Set axes labels
    ax.set_xlabel('Date',fontsize=20)
    ax.set_ylabel('Price',fontsize=20)
    ax.set_title('Testing Forecast Model Performance', fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()
    
def forecast_test(model, df, zipcode, n):
    # Getting a forecast for the next 36 months (3 years) after the last recorded date on our dataset.
    forecast = model.get_forecast(n)
    future_prediction = forecast.conf_int()
    future_prediction['value'] = forecast.predicted_mean
    future_prediction.columns = ['lower','upper','prediction'] 

    # Plotting our Forecast

    fig, ax = plt.subplots(figsize=(15, 7))
    df[zipcode].plot(ax=ax,label='Real Values')


    future_prediction['prediction'].plot(ax=ax,label='predicted value',ls='--')

    ax.fill_between(x= future_prediction.index, y1= future_prediction['lower'], 
                    y2= future_prediction['upper'],color='k',
                    label='Confidence Interval')
    ax.legend() 
    plt.ylabel("Price in $")
    plt.title(f'Average Home Price - {zipcode} - With Forcasted Value & Confidence Intervals')
    plt.grid()
    plt.show()
    
# forecasts into the future    
def forecast_future(model, df, zipcode):
    
    train = df[:'2016-01-01']
    test = df['2016-01-01':]
    
    pred = model.get_prediction(start=test.index.min(), 
          end=test.index.max(), dynamic = False)
    pred_conf = pred.conf_int()

    fig, ax = plt.subplots(figsize=(15, 7))

    plt.plot(train, label = 'Training Data')
    plt.plot(test, label = 'Test Data')
    plt.plot(model.predict(start=train.index.min(), 
              end=train.index.max())[1:], label = 'Train Prediction')
    # plt.plot(auto_arima.predict(start=test.index.min(), 
    #           end=test.index.max()), label = 'Test Prediction')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=0.5)
    ax.fill_between(pred_conf.index,
                    pred_conf.iloc[:, 0],
                    pred_conf.iloc[:, 1], color='k', alpha=0.5)
    plt.title('Average Home Price - 11216 - With Predicted & Confidence Intervals')
    plt.ylabel("Price in ($)")
    plt.title(f'Average Home Price - {zipcode} - With Forcasted Value & Confidence Intervals')
    plt.grid()
    plt.legend()