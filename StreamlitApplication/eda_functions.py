
import matplotlib.pyplot as plt
import streamlit as st
from pylab import rcParams
import seaborn as sns

def stationarity_test(dataframe, selected_date_column):
    timeseries = dataframe.set_index(selected_date_column).copy()
    # Get rolling statistics for window = 12 i.e. yearly statistics
    rolling_mean = timeseries.rolling(window = 3).mean()
    rolling_std = timeseries.rolling(window = 3).std()
    
    # Plot rolling statistic
    fig_stationarity = plt.figure(figsize= (10,6))
    plt.xlabel('Months')
    plt.ylabel('Sales Tax ')    
    plt.title('Stationary Test: Rolling Mean and Standard Deviation')
    plt.plot(timeseries, color= 'blue', label= 'Original')
    plt.plot(rolling_mean, color= 'green', label= 'Rolling Mean')
    plt.plot(rolling_std, color= 'red', label= 'Rolling Std')   
    plt.legend()
    st.pyplot(fig_stationarity)


def visualizing_pct_changes_monthly_yearly(dataframe, selected_date_column):
    dataframe_temp = dataframe.copy()
    dataframe_temp['Month'] = dataframe_temp[selected_date_column].dt.month

    month_mean_df = dataframe_temp.groupby('Month').sum().reset_index()
    month_mean_df.set_index('Month', drop=True, inplace=True)
    month_mean_df['Percentage_increase'] = month_mean_df.pct_change().fillna(0)*100


    dataframe_temp = dataframe_temp.drop('Month', axis =1)

    dataframe_temp['Year'] = dataframe_temp[selected_date_column].dt.year
    year_mean_df= dataframe_temp.groupby('Year').sum().reset_index()
    year_mean_df.set_index('Year', drop=True, inplace=True)

    year_mean_df['Percentage_increase']=year_mean_df.pct_change().fillna(0)*100
    fig, (ax1, ax2) = plt.subplots(2,figsize=(18,30))
    st.pyplot(fig)
    

# @st.cache 
def plot_seasonality(dataframe, selected_date_column, selected_seasonlaity_type):
    rcParams['figure.figsize'] = 20, 10

    dataframe = dataframe.set_index(selected_date_column)

    decomposition = sm.tsa.seasonal_decompose(dataframe, model = selected_seasonlaity_type)

    fig = decomposition.plot()
    st.pyplot(fig)
    
def box_plots(dataframe, selected_date_column, selected_amount_column):
    dataframe_temp = dataframe.copy()
    dataframe_temp['year'] = dataframe_temp[selected_date_column].dt.year
    dataframe_temp['month'] = dataframe_temp[selected_date_column].dt.month
    # years = dataframe_temp['year'].unique()

    #drawing the plot
    fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
    sns.boxplot(x= 'year', y= selected_amount_column, data = dataframe_temp, ax = axes[0])
    sns.boxplot(x= 'month', y= selected_amount_column, data = dataframe_temp, ax = axes[1])

    # Set Title
    axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
    axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    st.pyplot(fig)
