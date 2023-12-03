# streamlit
import pandas_datareader as pdr
import eda_functions

import streamlit as st
# data preprocessing and storage
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
# import numpy as np
# data viz
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
# from prophet import Prophet
# from neuralprophet import NeuralProphet
import numpy as np

import pre_processing_func
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn import metrics
import pmdarima as pm


#CSS configurations
st.set_page_config(layout = 'wide')
st.markdown(
    """
    <style>
    .main 
    background-color: #F5F5F5;
    </style>
    
    """,
    unsafe_allow_html = True
)

#https://arshren.medium.com/building-interactive-predictive-machine-learning-workflow-with-streamlit-330188c7ead0
#https://towardsdatascience.com/3-ways-to-create-a-multi-page-streamlit-app-1825b5b07c0f

## READ IN ALL THE DATA inot a pandas dataframe and make easy plot
#'FEDFUNDS' -> Federal funds rate, https://fred.stlouisfed.org/series/FEDFUNDS
# 'UNRATE' -> 'Unepmploymnt Rate' https://fred.stlouisfed.org/series/UNRATE
# 'MICH' -> Inflation Expectation https://fred.stlouisfed.org/series/MICH
# 'PSAVERT' -> Personal Saving Rate https://fred.stlouisfed.org/series/PSAVERT
# 'CIVPART' -> Labor Force Participation Rate ( https://fred.stlouisfed.org/series/CIVPART)
# 'MPRIME' -> Bank Prime Loan Rate https://fred.stlouisfed.org/series/MPRIME
def get_data():
    start = datetime(2005, 5, 1)
    end = datetime(2023, 10, 1)

    national_monthly_economic_indicators = pdr.DataReader(['FEDFUNDS', 'UNRATE', 'MICH', 'PSAVERT'],
                        'fred', start, end)

    national_monthly_economic_indicators.reset_index(inplace = True)
    national_monthly_economic_indicators.columns = ['Date', 'Federal_Funds_Rate', 'Unemployment_Rate', 'Inflation_Expectation', 'Personal_Savings_Rate']
    #Display urls to the data so user can get more infomation
    return national_monthly_economic_indicators
    
introduction = st.container()
landing_page_eda = st.container()
advanced_eda = st.container()
forecasting = st.container()


##HEADER
with introduction:
    st.header("Time series analysis Economic Indicators")
    st.text("This application was created to help guide users to perform time series analysis as well as forecasting using U.S Economic data is used provided by the Federal Bank of St. Louis.")
    st.subheader('Overview of the economic data')
# with landing_page_eda
    

with landing_page_eda:
    national_monthly_economic_indicators = get_data()
    #have user select a dates to filter out the data
    st.markdown('Select the dates to filter the data to the date ranges you want to explore')
    min_date = st.selectbox('Choose min date', pre_processing_func.dates_ascending(national_monthly_economic_indicators, 'Date'))
    economic_df_filtered = pre_processing_func.min_dates_filtering(national_monthly_economic_indicators,  'Date', min_date)
    max_date = st.selectbox('Choose max date', pre_processing_func.dates_descending(economic_df_filtered, 'Date'))
    economic_df_filtered = pre_processing_func.max_dates_filtering(economic_df_filtered,  'Date', max_date)
    

    #columns for the landing page
    display_economic_data, reference_data = st.columns(2)

    #diaplay the data in table format
    display_economic_data.dataframe(economic_df_filtered)
    
    #create a tbale to that users can access direct link to the data
    indicator_names = ['Federal Funds Rate', 'Unemployment Rate', 'Inflation Expectation', 'Personal Savings Rate','Personal Savings Rate']
    urls = ['https://fred.stlouisfed.org/series/FEDFUNDS', 'https://fred.stlouisfed.org/series/UNRATE', 'https://fred.stlouisfed.org/series/MICH', 'https://fred.stlouisfed.org/series/PSAVERT'] #, 'https://fred.stlouisfed.org/series/CIVPART', 'https://fred.stlouisfed.org/series/MPRIME']
    reference_table = pd.DataFrame(list(zip(indicator_names,urls )), 
                                   columns = ['Economic Indicator', 'Data Source'])
    reference_data.text('User Reference for data sources below:')
    reference_data.table(reference_table)
    
    
    
    
    #PLOTTING ALL THE INDICATORS ON THE SAME  graph
    basic_plot_all_indicators, correlation_plot = st.columns(2)
    fig_all_indicators = go.Figure()
    fig_all_indicators.add_trace(go.Scatter(
                    x=economic_df_filtered['Date'],
                    y=economic_df_filtered['Federal_Funds_Rate'],
                    name="Federal Funds Rate",
                    line_color='deepskyblue',
                    opacity=0.8))
    fig_all_indicators.add_trace(go.Scatter(
                    x=economic_df_filtered['Date'],
                    y=economic_df_filtered['Inflation_Expectation'],
                    name="Inflation Expectation",
                    line_color='red',
                    opacity=0.8))
    fig_all_indicators.add_trace(go.Scatter(
                    x=economic_df_filtered['Date'],
                    y=economic_df_filtered['Unemployment_Rate'],
                    name="Unemployment Rate",
                    line_color='green',
                    opacity=0.8))

    fig_all_indicators.add_trace(go.Scatter(
                    x=economic_df_filtered['Date'],
                    y=economic_df_filtered['Personal_Savings_Rate'],
                    name="Personal Savings Rate",
                    line_color='pink',
                    opacity=0.8))
    # Use date string to set xaxis range
    fig_all_indicators.update_layout(xaxis_range=[economic_df_filtered.iloc[0]['Date'],economic_df_filtered.iloc[-1]['Date']],
                    title_text="Monthly U.S Economic data over time",xaxis_rangeslider_visible=True)
    # fig.show()
    # fig_all_indicators.update_layout(template="simple_white", font=dict(size=18), title_text='Forecasting using Linear Regression model',
    #                                 width=650, title_x=0.5, height=400, xaxis_title='Date',
    #                                 yaxis_title='Percent')
    
    basic_plot_all_indicators.plotly_chart(fig_all_indicators)
    basic_plot_all_indicators.caption('Plotting all of the economic indicators on the same chart for comparison. Use the slider to focus on a certain time period')

        
    
    correlation = economic_df_filtered.drop('Date', axis = 1).corr().round(2)
    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Viz
    df_corr_viz = correlation.mask(mask)#.dropna(how='all').dropna('columns', how='all')
    fig_correlation = px.imshow(df_corr_viz, text_auto=True)
    correlation_plot.caption('This is a correlation plot among all the different economic indicators')

    correlation_plot.plotly_chart(fig_correlation)
    
    #giving user the option to choose between exploring EDA or performing forecasting
    perform_advanced_eda = st.checkbox('Click to explore an economic indicator further')
    perform_forecasting = st.checkbox('Click to perform forecasting on an economic indicator')
with advanced_eda:
    
    #Have user choose the indicator that they want to explore further
    if perform_advanced_eda:
        
        st.header('Exploratory Data Analysis')
        indicator_options = list(economic_df_filtered.columns)
        indicator_options.remove('Date')
        user_selected_indicator = st.selectbox('Choose economic indicator to explore furter', indicator_options)


        #once an economic indicator is selected we should prompt the user to either analyze the data further or explore forecasting
        economic_df_single_indic = economic_df_filtered[['Date', user_selected_indicator]]
        if user_selected_indicator:
            st.text('Displaying selected economic indicator')
            st.dataframe(economic_df_single_indic)
        
            bar_chart, line_chart = st.columns(2)
         
            #BAr chart
            bar_chart.caption('Bar chart of economic indicator rate over time with top 3 highest value points indicated')
            top_dates = economic_df_single_indic.sort_values(by=[user_selected_indicator],ascending=False).head(3)
            vals = []
            for tgl, tot in zip(top_dates['Date'], top_dates[user_selected_indicator]):
                tgl = tgl.strftime("%d %B")
                val = "%d (%s)"%(tot, tgl)
                vals.append(val)
            top_dates['tgl'] = vals
                            # top_dates

            fig_w_traces = go.Figure(data=go.Bar(x = economic_df_single_indic['Date'].astype(dtype=str), 
                                                                y = economic_df_single_indic[user_selected_indicator],
                                                                marker_color='black', text=user_selected_indicator))
            fig_w_traces.add_traces(go.Scatter(x=top_dates['Date'], y=top_dates[user_selected_indicator],
                                                        textposition='top left',
                                                        textfont=dict(color='#233a77'),
                                                        mode='markers+text',
                                                        marker=dict(color='red', size=6),
                                                        text = top_dates["tgl"]))
                                    
                                # fig_w_traces.show()
            bar_chart.plotly_chart(fig_w_traces)
            
            
            #Line chart
            line_chart.caption('Line chart with top 3 values in selected date range highlighted')
            top_dates = economic_df_single_indic.sort_values(by=[user_selected_indicator],ascending=False).head(3)
            vals = []
            for tgl, tot in zip(top_dates['Date'], top_dates[user_selected_indicator]):
                tgl = tgl.strftime("%d %B")
                val = "%d (%s)"%(tot, tgl)
                vals.append(val)
            top_dates['tgl'] = vals
                        # top_dates

            fig_w_traces = go.Figure(data=go.Scatter(x = economic_df_single_indic['Date'].astype(dtype=str), 
                                                            y = economic_df_single_indic[user_selected_indicator],
                                                            marker_color='black', text=user_selected_indicator))

            fig_w_traces.add_traces(go.Scatter(x=top_dates['Date'], y=top_dates[user_selected_indicator],
                                                    textposition='top left',
                                                    textfont=dict(color='#233a77'),
                                                    mode='markers+text',
                                                    marker=dict(color='red', size=6),
                                                    text = top_dates["tgl"]))
                                
                            # fig_w_traces.show()
            fig_w_traces.update_layout(template="simple_white", font=dict(size=18), title_text='Forecasting using Linear Regression model',
                                width=650, title_x=0.5, height=400, xaxis_title='Date',
                                yaxis_title='Percent')
            
            line_chart.plotly_chart(fig_w_traces)
            

            viz_seasonality, smoothed_trend = st.columns(2)
            
            dataframe_temp = economic_df_single_indic.copy()
            dataframe_temp['year'] = dataframe_temp['Date'].dt.year
            dataframe_temp['month'] = dataframe_temp['Date'].dt.month
            # years = dataframe_temp['year'].unique()

            #drawing the plot
            fig_seasonality_boxplots, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
            sns.boxplot(x= 'year', y= user_selected_indicator, data = dataframe_temp, ax = axes[0])
            sns.boxplot(x= 'month', y= user_selected_indicator, data = dataframe_temp, ax = axes[1])

            # Set Title
            axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
            axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
            viz_seasonality.text('Boxplots of our economic indicator')
            viz_seasonality.caption('Plotting the data by month to show Seasonality as well as by year to show the overall Trend')
            viz_seasonality.pyplot(fig_seasonality_boxplots)
            
                

            # #Showing smoothed data
            # smoothen_data = smoothed_trend.checkbox('View smoothened data')
            # if smoothen_data:
                
            smoothed_trend.subheader('Smoothing the data with a rolling average')
            smoothed_trend.caption('Smoothing the data using rolling mean helps us to visualize the trend over time')
            window_options = [3, 6, 9, 12]
            window_selection = smoothed_trend.selectbox('Choose a window to use for the rolling average)', window_options)

            timeseries = economic_df_single_indic.set_index('Date').copy()
                            # Get rolling statistics for window = 12 i.e. yearly statistics
            rolling_mean = timeseries.rolling(window = window_selection).mean()
            rolling_std = timeseries.rolling(window = window_selection).std()
                            
                            # Plot rolling statistic
            fig_stationarity = plt.figure(figsize= (10,6))
            plt.xlabel('Months')
            plt.ylabel('Percent')    
            plt.title('Rolling Mean and Standard Deviation')
            plt.plot(timeseries, color= 'blue', label= 'Original')
            plt.plot(rolling_mean, color= 'green', label= 'Rolling Mean')
            plt.plot(rolling_std, color= 'red', label= 'Rolling Std')   
            plt.legend()
            smoothed_trend.pyplot(fig_stationarity) 
                # smoothed_trend.write(rolling_mean) 

with forecasting:
    
    if perform_forecasting:
       
        
        st.header('Forecast data')
        st.text('Here we can perform forecasting to see how well our selected model can accurately forecast our selected indicator')
        # st.text('We will be using a machine learning model made by Facebook called prophet to perform forecating')
        # st.markdown('Choose an economic indicator to perform forecasting')
        model_config_training, accuracy = st.columns(2)

        indicator_options_forecasting = list(economic_df_filtered.columns)
        indicator_options_forecasting.remove('Date')
        
        #Model Configuration textbox
        model_config_training.subheader('Model Configuration')
        user_selected_indicator_forecasting = model_config_training.selectbox('Choose economic indicator to explore furter', indicator_options_forecasting)


        #once an economic indicator is selected we should prompt the user to either analyze the data further or explore forecasting
        economic_df_single_indic = economic_df_filtered[['Date', user_selected_indicator_forecasting]]
       
        model_config_training.text('The dates selected for training is what is used to train the model, and the testing period is used to compare our predicted values to our actual data')
        economic_df_single_indic.reset_index(drop = True, inplace = True)
        index_vals = list(economic_df_single_indic.index.values)
        default_val = int(index_vals[-12])
        
        cutoff_date = model_config_training.selectbox('Choose cut off date for training/validation. If not sure keep default value.', economic_df_single_indic['Date'].sort_values().unique(), index = default_val, help = "Usually we would reccommend an 80/20 split for train/test or the last 12 months for testing.")
        dataframe_modeling = economic_df_single_indic.copy()

        mask1 = (dataframe_modeling['Date'] <= cutoff_date)
        mask2 = (dataframe_modeling['Date'] > cutoff_date)

        df_tr = dataframe_modeling[mask1]
        df_tst = dataframe_modeling[mask2]

        training_df, testing_df = st.columns(2)
        
        model_config_training.text('Visualize our train and test data')                    
        with training_df:
            model_config_training.write('Dataframe used for training the model')
            model_config_training.write(df_tr)
        with testing_df:
            model_config_training.write('Dataframe used for testing the model')
            model_config_training.write(df_tst)
            X_tst_length = df_tst.shape[0]
        
        
        
        
        
        #Here we want to give the user to choose either Linear Regression, Exponential Smoothing or both
        #
        model_type = model_config_training.selectbox('Choose the model we want to use for forecasting!', ['Linear Regression', 'Exponential Smoothing', 'Auto-Arima'])
        
        show_forecasts = st.checkbox('Click to show forecasts after configuring the model')    
        if show_forecasts:
            with accuracy:
                st.subheader('Accuracy Metrics')
                if model_type == 'Linear Regression':
                #get the X_train and y_train
                    lin_reg_df = dataframe_modeling.copy()
                    lin_reg_df['Time'] = np.arange(len(lin_reg_df.index))
                    lin_reg_df.drop('Date', axis = 1, inplace = True)
                    df_tr_lin = lin_reg_df[mask1]
                    df_tst_lin = lin_reg_df[mask2]
                    
                    
                    X_train = df_tr_lin['Time']
                    X_train = np.array(X_train).reshape(-1, 1)
                    
                    y_train = df_tr_lin[user_selected_indicator_forecasting]
                    X_test = df_tst_lin['Time']
                    X_test = np.array(X_test).reshape(-1, 1)
                    y_test = df_tst_lin[user_selected_indicator_forecasting]
        

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                
                    y_pred =  list(model.predict(X_test))
                    
                    results_df = pd.DataFrame({'Actuals':y_test, 'Predicted Values': y_pred })
                    
                    
                

                    
                    mae = metrics.mean_absolute_error(y_test,y_pred)
                    mse = metrics.mean_squared_error(y_test,y_pred)
                    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    mape *=100
                    
                    acc_metrics = ['Mean Absolute Error', 'Mean squared Error', 'Root Mean Squared Error', 'Mean Absolute Percentage Error']
                    acc_vals = [mae,mse, rmse, mape ]
                    acc_metrics_df  = pd.DataFrame(list(zip(acc_metrics, acc_vals)), columns =['Accuracy Metric', 'Value']
                    )
                    st.text('Actual values compared to forecasted values')
                    st.write(results_df)

                    st.write('Overview of Accuracy Metrics')
                    st.write(acc_metrics_df)
                    
                    X_test_plotting = pd.DataFrame()
                    X_test_plotting['Actuals'] = y_test
                    X_test_plotting['Date'] = df_tst['Date'].values
                    # st.write(X_test_plotting)
                    
                    X_test_forecast = pd.DataFrame()
                    X_test_forecast['Forecast'] = y_pred
                    X_test_forecast['Date'] = df_tst['Date'].values
                    # st.write(X_test_forecast)
                    fig_lin_results = go.Figure()
                    fig_lin_results.add_trace(go.Scatter(x=df_tr['Date'], y=df_tr[user_selected_indicator_forecasting], name='Train'))
                    fig_lin_results.add_trace(go.Scatter(x=df_tst['Date'], y=df_tst[user_selected_indicator_forecasting], name='Test'))
                    fig_lin_results.add_trace(go.Scatter(x=X_test_forecast['Date'], y=X_test_forecast['Forecast'], name='Linear Regression'))
                    fig_lin_results.update_layout(template="simple_white", font=dict(size=18), title_text='Forecasting using Linear Regression model',
                                    width=650, title_x=0.5, height=400, xaxis_title='Date',
                                    yaxis_title='Percent')
                    st.plotly_chart(fig_lin_results)

                
            #If the user clicks on exponential smoothing
                elif model_type == 'Exponential Smoothing':
                    model_holt = Holt(df_tr[user_selected_indicator_forecasting], damped_trend=True).fit(optimized=True)
                    forecasts_holt = model_holt.forecast(len(df_tst))
                    forecasts_holt_df = pd.DataFrame(
                                    {
                                        'ds': df_tst['Date'],
                                        'yhat': forecasts_holt,
                                    }
                                )
                    
                    #Calculate the accuracy metrics
                    y_test = df_tst[user_selected_indicator_forecasting].values


                    results_df_holt = pd.DataFrame({'Actuals':y_test, 'Predicted Values': forecasts_holt })
                    st.text('Actual values compared to forecasted values')

                    st.write(results_df_holt)
                    
                    mae = metrics.mean_absolute_error(y_test,forecasts_holt)
                    mse = metrics.mean_squared_error(y_test,forecasts_holt)
                    rmse = np.sqrt(metrics.mean_squared_error(y_test, forecasts_holt))
                    mape = mean_absolute_percentage_error(y_test, forecasts_holt)
                    mape *=100

                    acc_metrics = ['Mean Absolute Error', 'Mean squared Error', 'Root Mean Squared Error', 'Mean Absolute Percentage Error']
                    acc_vals = [mae,mse, rmse, mape ]
                    acc_metrics_df_holt  = pd.DataFrame(list(zip(acc_metrics, acc_vals)), columns =['Accuracy Metric', 'Value']
                    )
                    
                    st.write('Overview of Accuracy Metrics')
                    st.write(acc_metrics_df_holt)
                    
                    
                    #Plot the forecast vs actuals
                    fig_exp = go.Figure()
                    fig_exp.add_trace(go.Scatter(x=df_tr['Date'], y=df_tr[user_selected_indicator_forecasting], name='Train'))
                    fig_exp.add_trace(go.Scatter(x=df_tst['Date'], y=df_tst[user_selected_indicator_forecasting], name='Test'))
                    fig_exp.add_trace(go.Scatter(x=forecasts_holt_df['ds'], y=forecasts_holt_df['yhat'], name='Exp_Smoothing'))
                    fig_exp.update_layout(template="simple_white", font=dict(size=18), title_text='Forecasting using Exponential smoothing model',
                                    width=650, title_x=0.5, height=400, xaxis_title='Date',
                                    yaxis_title='Percent')
                    st.plotly_chart(fig_exp)
                elif model_type == 'Auto-Arima':
                    df_tr_arima = df_tr.copy()
                    df_tr_arima.set_index('Date', inplace = True)
                    arima_model = pm.auto_arima(df_tr_arima, seasonal=True, m=12)
                    aima_forecast_df = pd.DataFrame(arima_model.predict(X_tst_length), index = df_tst['Date'])
                    aima_forecast_df.columns =['Forecast']
                    aima_forecast_df.reset_index(inplace = True)
                    # st.write(aima_forecast_df)
                    
                    y_test = df_tst[user_selected_indicator_forecasting].values
                    predicted_vals = aima_forecast_df['Forecast'].values
                    results_df_auto_arima = pd.DataFrame({'Actuals':y_test, 'Predicted Values': predicted_vals })
                    st.text('Actual values compared to forecasted values')
                    st.write(results_df_auto_arima)
                    
                    
                    mae = metrics.mean_absolute_error(y_test,predicted_vals)
                    mse = metrics.mean_squared_error(y_test,predicted_vals)
                    rmse = np.sqrt(metrics.mean_squared_error(y_test, predicted_vals))
                    mape = mean_absolute_percentage_error(y_test, predicted_vals)
                    mape *=100

                    acc_metrics = ['Mean Absolute Error', 'Mean squared Error', 'Root Mean Squared Error', 'Mean Absolute Percentage Error']
                    acc_vals = [mae,mse, rmse, mape ]
                    acc_metrics_df_arima  = pd.DataFrame(list(zip(acc_metrics, acc_vals)), columns =['Accuracy Metric', 'Value']
                    )
                    st.write('Overview of Accuracy Metrics')
                    st.write(acc_metrics_df_arima)
                    
                    
                    
                    
                    
                    
                    fig_arima = go.Figure()
                    fig_arima.add_trace(go.Scatter(x=df_tr['Date'], y=df_tr[user_selected_indicator_forecasting], name='Train'))
                    fig_arima.add_trace(go.Scatter(x=df_tst['Date'], y=df_tst[user_selected_indicator_forecasting], name='Test'))
                    fig_arima.add_trace(go.Scatter(x=aima_forecast_df['Date'], y=aima_forecast_df['Forecast'], name='Auto-Arima'))
                    fig_arima.update_layout(template="simple_white", font=dict(size=18), title_text='Forecasting using Auto-Arima',
                                    width=650, title_x=0.5, height=400, xaxis_title='Date',
                                    yaxis_title='Percent')
                    st.plotly_chart(fig_arima)
                    
                    

                    
