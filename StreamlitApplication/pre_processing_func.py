import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
# import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.pyplot as plt

import datetime as dt
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt




def min_dates_filtering(dataframe, selected_date_column, min_date):
    dataframe = dataframe[dataframe[selected_date_column] >= min_date]
    return dataframe

def max_dates_filtering(dataframe, selected_date_column, max_date):
    dataframe = dataframe[dataframe[selected_date_column] <= max_date]
    return dataframe


def dates_descending(dataframe, selected_date_column):
    date_list_descending = dataframe[selected_date_column].sort_values(ascending = False).unique()
    return date_list_descending

def dates_ascending(dataframe, selected_date_column):
    date_list_ascending = dataframe[selected_date_column].sort_values().unique()
    return date_list_ascending
