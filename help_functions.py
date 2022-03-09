import pandas as pd
import pandas.tseries
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from pandas.plotting import autocorrelation_plot,lag_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import autocorrelation_plot,lag_plot
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import rcParams

import itertools
import warnings

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
    will be the mean of the values from the datetime columns in all of the rows.
    """
    
    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 
                                  'State', 'Metro', 'CountyName', 'ROI'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted #.groupby('time').aggregate({'value':'mean'})


def plot_trend(data, title):
    data.plot.line(color = 'green')
    plt.title(title)
    plt.ylabel('Price ($ in Thousands)')
    plt.grid()
    plt.show()
    
def plot_mean(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(8,6))
    sns.barplot(x,y, palette = 'Blues_d')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.grid()
    plt.title(f'{title} in New York City')

    
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)

# NOTE: if you visualizations are too cluttered to read, try calling 'plt.gcf().autofmt_xdate()'!

def make_plot_count(col, data, order = None):
    sns.countplot(x = col, data = data, palette = 'icefire_r', order = order)
    plt.title(f'Frequency in {col}')
    plt.show()
    
def plot_mean(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(8,6))
    sns.barplot(x,y, palette = 'Blues_d')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.title(f'{title} in New York City')
