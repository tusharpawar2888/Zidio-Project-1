import streamlit as st

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(layout="wide", page_title="Stock Market Analysis")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

st.title("Stock Market Data Analytics")

# Display warnings for missing optional dependencies
if not PROPHET_AVAILABLE:
    st.warning("Prophet not installed. Prophet forecasting will be unavailable.")
if not LSTM_AVAILABLE:
    st.warning("TensorFlow/Scikit-learn not installed. LSTM forecasting will be unavailable.")

# File upload section
uploaded_file = st.file_uploader("Upload your stock market CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Overview")
    st.write("Dataset Shape:", df.shape)
    
    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df.head())
    
    # Data cleaning section
    st.subheader("Data Cleaning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display info before cleaning
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text("Before Cleaning:")
        st.text(buffer.getvalue())
    
    # Data cleaning operations
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Drop rows where 'Date' is NaT (conversion failed)
    df = df.dropna(subset=['Date'])
    
    # Fill missing values in 'Trades' column with the median
    if 'Trades' in df.columns:
        df['Trades'] = df['Trades'].fillna(df['Trades'].median())
    
    # Fill missing values in numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    with col2:
        # Display info after cleaning
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text("After Cleaning:")
        st.text(buffer.getvalue())
    
    # Basic Visualizations
    st.subheader("Basic Visualizations")
    
    # Stock symbols filter
    if 'Symbol' in df.columns:
        symbols = df['Symbol'].unique().tolist()
        selected_symbol = st.selectbox("Select Stock Symbol", symbols)
        df_filtered = df[df['Symbol'] == selected_symbol]
    else:
        df_filtered = df
        st.warning("No 'Symbol' column found in the data. Using all data for analysis.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Trends", "Volume Analysis", "Correlation", "Returns"])
    
    with tab1:
        st.subheader("Stock Price Trends")
        
        fig1 = px.line(df_filtered, x='Date', y='Close', title='Stock Closing Prices Over Time')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Monthly average close price
        df_filtered['Month'] = df_filtered['Date'].dt.to_period('M')
        monthly_avg = df_filtered.groupby('Month')['Close'].mean().reset_index()
        monthly_avg['Month'] = monthly_avg['Month'].astype(str)
        
        fig2 = px.line(monthly_avg, x='Month', y='Close', title='Monthly Average Close Price')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Candlestick chart
        fig3 = go.Figure(data=[go.Candlestick(
            x=df_filtered['Date'],
            open=df_filtered['Open'],
            high=df_filtered['High'],
            low=df_filtered['Low'],
            close=df_filtered['Close']
        )])
        fig3.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.subheader("Trading Volume Analysis")
        
        fig4 = px.line(df_filtered, x='Date', y='Volume', title='Trading Volume Over Time')
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Correlation heatmap
        corr = df_filtered.corr(numeric_only=True)
        fig5 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                         title='Correlation Between Variables')
        st.plotly_chart(fig5, use_container_width=True)
        
        # Boxplot
        fig6 = px.box(df_filtered, y=['Open', 'High', 'Low', 'Close'], title='Boxplot of Stock Prices')
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab4:
        st.subheader("Returns Analysis")
        
        # Calculate daily returns
        df_filtered['Daily Return'] = df_filtered['Close'].pct_change()
        
        fig7 = px.histogram(df_filtered.dropna(), x='Daily Return', nbins=50,
                           title='Distribution of Daily Returns',
                           marginal='box')
        st.plotly_chart(fig7, use_container_width=True)
    
    # Time Series Forecasting
    st.subheader("Time Series Forecasting")
    
    # Check if we have enough data for time series analysis
    if len(df_filtered) < 30:
        st.warning("Not enough data for reliable time series forecasting. Please upload more data (at least 30 data points recommended).")
        st.stop()
    
    # Prepare time series data
    ts_data = df_filtered[['Date', 'Close']].sort_values('Date').copy()
    ts_data.set_index('Date', inplace=True)
    
    # Handle missing values in time series
    ts_data['Close'] = ts_data['Close'].interpolate(method='linear')
    
    # Remove any remaining NaN values
    ts_data = ts_data.dropna()
    
    # Display time series summary
    st.subheader("Time Series Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(ts_data))
    with col2:
        try:
            date_range = f"{ts_data.index.min().strftime('%Y-%m-%d')} to {ts_data.index.max().strftime('%Y-%m-%d')}"
        except:
            date_range = f"{str(ts_data.index.min())[:10]} to {str(ts_data.index.max())[:10]}"
        st.metric("Date Range", date_range)
    with col3:
        st.metric("Min Price", f"${ts_data['Close'].min():.2f}")
    with col4:
        st.metric("Max Price", f"${ts_data['Close'].max():.2f}")
    
    # Test for stationarity
    result = adfuller(ts_data['Close'])
    stationarity_text = f"""
    ADF Statistic: {result[0]:.4f}
    p-value: {result[1]:.4f}
    """
    st.code(stationarity_text)
    
    if result[1] > 0.05:
        st.warning("The time series is not stationary. Differencing will be applied.")
        # Differencing
        ts_data_diff = ts_data['Close'].diff().dropna()
        
        # Test differenced series
        result_diff = adfuller(ts_data_diff)
        diff_text = f"""
        ADF Statistic (Differenced): {result_diff[0]:.4f}
        p-value (Differenced): {result_diff[1]:.4f}
        """
        st.code(diff_text)
    else:
        st.success("The time series is stationary.")
        ts_data_diff = ts_data['Close']
    
    # ACF and PACF plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ACF Plot")
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            max_lags = min(30, len(ts_data_diff) // 4)  # Ensure we don't exceed data length
            plot_acf(ts_data_diff, ax=ax, lags=max_lags)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting ACF: {str(e)}")
    
    with col2:
        st.subheader("PACF Plot")
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            max_lags = min(30, len(ts_data_diff) // 4)  # Ensure we don't exceed data length
            plot_pacf(ts_data_diff, ax=ax, lags=max_lags)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error plotting PACF: {str(e)}")
    
    # Forecasting section
    st.subheader("Forecasting")
    
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=1)
    
    # Create tabs for different forecasting models
    tabs = ["ARIMA Model", "SARIMA Model"]
    if PROPHET_AVAILABLE:
        tabs.append("Prophet Model")
    if LSTM_AVAILABLE:
        tabs.append("LSTM Model")
    
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:
        # ARIMA parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1, step=1)
        with col2:
            d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=1, step=1)
        with col3:
            q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1, step=1)
        
        if st.button("Run ARIMA Forecast"):
            with st.spinner("Running ARIMA model..."):
                try:
                    # Fit ARIMA model
                    model = ARIMA(ts_data['Close'], order=(p, d, q))
                    model_fit = model.fit()
                    
                    # Get summary
                    st.text(model_fit.summary())
                    
                    # Make forecast
                    forecast = model_fit.forecast(steps=forecast_days)
                    
                    # Create forecast dataframe
                    last_date = ts_data.index[-1]
                    forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
                    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)
                    
                    # Plot results
                    fig = go.Figure()
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=ts_data.index, y=ts_data['Close'],
                        mode='lines', name='Historical'
                    ))
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index, y=forecast_df['Forecast'],
                        mode='lines', name='Forecast', line=dict(dash='dash')
                    ))
                    fig.update_layout(title=f"ARIMA({p},{d},{q}) Forecast", xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in ARIMA model: {str(e)}")
    
    with tab_objects[1]:
        # SARIMA parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("AR order (p) for SARIMA", min_value=0, max_value=5, value=1, step=1)
            P = st.number_input("Seasonal AR order (P)", min_value=0, max_value=2, value=1, step=1)
        with col2:
            d = st.number_input("Differencing order (d) for SARIMA", min_value=0, max_value=2, value=1, step=1)
            D = st.number_input("Seasonal differencing (D)", min_value=0, max_value=1, value=1, step=1)
        with col3:
            q = st.number_input("MA order (q) for SARIMA", min_value=0, max_value=5, value=1, step=1)
            Q = st.number_input("Seasonal MA order (Q)", min_value=0, max_value=2, value=1, step=1)
        
        seasonality = st.selectbox("Seasonality period", options=[7, 12, 30, 52], index=0, 
                                 help="7=Weekly, 12=Monthly for monthly data, 30=Monthly for daily data, 52=Yearly for weekly data")
        
        if st.button("Run SARIMA Forecast"):
            with st.spinner("Running SARIMA model..."):
                try:
                    # Fix index for SARIMA
                    ts_data_fixed = ts_data.copy()
                    ts_data_fixed.index = pd.DatetimeIndex(ts_data_fixed.index)
                    
                    # Ensure we have enough data for the seasonal component
                    if len(ts_data_fixed) < seasonality * 2:
                        st.warning(f"Not enough data for seasonality period {seasonality}. Consider using a smaller period or more data.")
                        seasonality = min(seasonality, len(ts_data_fixed) // 2)
                    
                    # Fit SARIMA model
                    model = SARIMAX(ts_data_fixed['Close'],
                                   order=(p, d, q),
                                   seasonal_order=(P, D, Q, seasonality))
                    model_fit = model.fit(disp=False, maxiter=100)
                    
                    # Get summary
                    st.text(model_fit.summary())
                    
                    # Make forecast
                    forecast = model_fit.get_forecast(steps=forecast_days)
                    forecast_mean = forecast.predicted_mean
                    conf_int = forecast.conf_int()
                    
                    # Create forecast dataframe
                    last_date = ts_data.index[-1]
                    forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
                    
                    # Plot results
                    fig = go.Figure()
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=ts_data.index, y=ts_data['Close'],
                        mode='lines', name='Historical'
                    ))
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_dates, y=forecast_mean,
                        mode='lines', name='Forecast', line=dict(dash='dash')
                    ))
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
                        fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval'
                    ))
                    fig.update_layout(title=f"SARIMA({p},{d},{q})({P},{D},{Q},{seasonality}) Forecast", 
                                     xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in SARIMA model: {str(e)}")
    
    # Prophet Model Tab
    if PROPHET_AVAILABLE and len(tab_objects) > 2:
        with tab_objects[2]:
            st.subheader("Prophet Forecasting")
            
            if st.button("Run Prophet Forecast"):
                with st.spinner("Running Prophet model..."):
                    try:
                        # Prepare data for Prophet
                        df_prophet = ts_data.reset_index()[['Date', 'Close']].copy()
                        df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
                        
                        # Fit Prophet model
                        model_prophet = Prophet(
                            daily_seasonality=True,
                            weekly_seasonality=True,
                            yearly_seasonality=True
                        )
                        model_prophet.fit(df_prophet)
                        
                        # Make future dataframe
                        future = model_prophet.make_future_dataframe(periods=forecast_days)
                        forecast_prophet = model_prophet.predict(future)
                        
                        # Plot results
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=df_prophet['ds'], y=df_prophet['y'],
                            mode='lines', name='Historical', line=dict(color='blue')
                        ))
                        
                        # Forecast
                        forecast_future = forecast_prophet[forecast_prophet['ds'] > df_prophet['ds'].max()]
                        fig.add_trace(go.Scatter(
                            x=forecast_future['ds'], y=forecast_future['yhat'],
                            mode='lines', name='Forecast', line=dict(color='red', dash='dash')
                        ))
                        
                        # Confidence intervals
                        fig.add_trace(go.Scatter(
                            x=forecast_future['ds'].tolist() + forecast_future['ds'].tolist()[::-1],
                            y=forecast_future['yhat_upper'].tolist() + forecast_future['yhat_lower'].tolist()[::-1],
                            fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title='Prophet Forecast',
                            xaxis_title='Date',
                            yaxis_title='Price'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast components
                        st.subheader("Forecast Components")
                        fig_components = model_prophet.plot_components(forecast_prophet)
                        st.pyplot(fig_components)
                        
                    except Exception as e:
                        st.error(f"Error in Prophet model: {str(e)}")
    
    # LSTM Model Tab
    if LSTM_AVAILABLE and len(tab_objects) > (3 if PROPHET_AVAILABLE else 2):
        tab_index = 3 if PROPHET_AVAILABLE else 2
        with tab_objects[tab_index]:
            st.subheader("LSTM Neural Network Forecasting")
            
            # LSTM parameters
            col1, col2 = st.columns(2)
            with col1:
                time_steps = st.number_input("Time Steps (lookback window)", min_value=10, max_value=100, value=60, step=5)
                epochs = st.number_input("Training Epochs", min_value=10, max_value=100, value=20, step=5)
            with col2:
                lstm_units = st.number_input("LSTM Units", min_value=10, max_value=200, value=50, step=10)
                batch_size = st.number_input("Batch Size", min_value=16, max_value=128, value=32, step=16)
            
            if st.button("Run LSTM Forecast"):
                with st.spinner("Training LSTM model... This may take a few minutes."):
                    try:
                        # Prepare data for LSTM
                        close_data = ts_data['Close'].values.reshape(-1, 1)
                        
                        # Scale the data
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(close_data)
                        
                        # Create sequences
                        def create_sequences(data, time_steps):
                            X, y = [], []
                            for i in range(time_steps, len(data)):
                                X.append(data[i-time_steps:i])
                                y.append(data[i])
                            return np.array(X), np.array(y)
                        
                        X, y = create_sequences(scaled_data, time_steps)
                        
                        # Split data
                        split = int(0.8 * len(X))
                        X_train, X_test = X[:split], X[split:]
                        y_train, y_test = y[:split], y[split:]
                        
                        # Build LSTM model
                        model_lstm = Sequential([
                            LSTM(units=lstm_units, return_sequences=True, input_shape=(time_steps, 1)),
                            Dropout(0.2),
                            LSTM(units=lstm_units),
                            Dropout(0.2),
                            Dense(units=1)
                        ])
                        
                        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                        
                        # Train model
                        history = model_lstm.fit(
                            X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            verbose=0
                        )
                        
                        # Make predictions on test set
                        test_predictions = model_lstm.predict(X_test)
                        test_predictions = scaler.inverse_transform(test_predictions)
                        actual_test = scaler.inverse_transform(y_test)
                        
                        # Forecast future values
                        last_sequence = scaled_data[-time_steps:]
                        future_predictions = []
                        
                        for _ in range(forecast_days):
                            next_pred = model_lstm.predict(last_sequence.reshape(1, time_steps, 1), verbose=0)
                            future_predictions.append(next_pred[0, 0])
                            last_sequence = np.append(last_sequence[1:], next_pred)
                        
                        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                        
                        # Create forecast dates
                        last_date = ts_data.index[-1]
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                        
                        # Plot results
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=ts_data.index, y=ts_data['Close'],
                            mode='lines', name='Historical', line=dict(color='blue')
                        ))
                        
                        # Test predictions
                        test_dates = ts_data.index[split+time_steps:]
                        fig.add_trace(go.Scatter(
                            x=test_dates, y=test_predictions.flatten(),
                            mode='lines', name='Test Predictions', line=dict(color='orange')
                        ))
                        
                        # Future forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_dates, y=future_predictions.flatten(),
                            mode='lines', name='Forecast', line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title='LSTM Forecast',
                            xaxis_title='Date',
                            yaxis_title='Price'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show training history
                        st.subheader("Training History")
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                        fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                        fig_loss.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in LSTM model: {str(e)}")

else:
    st.info("Please upload a CSV file to begin analysis.")
    
    # Sample data format
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV file should have columns similar to:
    - Date: Date of stock trading
    - Open: Opening price
    - High: Highest price of the day
    - Low: Lowest price of the day
    - Close: Closing price
    - Volume: Trading volume
    - Symbol (optional): Stock symbol if multiple stocks are in the dataset
    """)