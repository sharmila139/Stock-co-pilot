import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set page configuration
st.set_page_config(layout="wide", page_title="Stock Dashboard")

# Sidebar
def create_sidebar():
    st.sidebar.title("Stock Dashboard")
    
    # Default stock list
    default_stocks = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'META': 'Meta Platforms Inc.',
        'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corporation',
        'JPM': 'JPMorgan Chase & Co.',
        'BAC': 'Bank of America Corp.',
        'DIS': 'The Walt Disney Company',
        'INTC': 'Intel Corporation
    }
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select a stock:",
        list(default_stocks.keys()),
        format_func=lambda x: f"{x} - {default_stocks[x]}"
    )
    
    return selected_stock

# Function to get stock data
def get_stock_data(ticker, period='2y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    info = stock.info
    return data, info

# Function to create LSTM model
def create_lstm_model(data, prediction_days=90):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    x_train = []
    y_train = []
    print(len(scaled_data))
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(10),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
    
    return model, scaler

# Function to make predictions
def predict_future(model, data, scaler, prediction_days=60, future_days=30):
    total_dataset = data['Close'].values.reshape(-1, 1)
    model_inputs = total_dataset[len(total_dataset) - prediction_days:]
    model_inputs = scaler.transform(model_inputs)
    
    predictions = []
    
    for _ in range(future_days):
        x_test = model_inputs[-prediction_days:]
        x_test = np.reshape(x_test, (1, prediction_days, 1))
        
        pred = model.predict(x_test, verbose=0)
        predictions.append(pred[0, 0])
        
        model_inputs = np.append(model_inputs, pred)
        model_inputs = np.delete(model_inputs, 0)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Main application
def main():
    # Create sidebar and get selected stock
    selected_stock = create_sidebar()
    
    # Time period selection
    time_periods = {
        '1W': '7d',
        '1M': '1mo',
        '3M': '3mo',
        '6M': '6mo',
        'YTD': 'ytd',
        '1Y': '1y',
        '2Y': '2y',
        '5Y': '5y'
    }
    
    selected_period = st.selectbox('Select Time Period:', list(time_periods.keys()))
    
    # Get stock data
    data, info = get_stock_data(selected_stock, time_periods[selected_period])
    
    # Display stock info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.title(f"{info['longName']} ({selected_stock})")
        current_price = data['Close'].iloc[-1]
        price_change = current_price - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
        st.metric("Volume", f"{info.get('volume', 0):,}")
    
    with col3:
        st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
        st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")
    
    # Create stock price chart
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))
    
    fig.update_layout(
        title=f'{selected_stock} Stock Price',
        yaxis_title='Stock Price (USD)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction section
    st.subheader("Price Prediction")
    prediction_days = st.slider("Select number of days for prediction:", 1, 90, 30)
    
    if st.button("Generate Prediction"):
        with st.spinner("Training model and generating predictions..."):
            model, scaler = create_lstm_model(data)
            predictions = predict_future(model, data, scaler, 60, prediction_days)
            
            # Create prediction dates
            last_date = data.index[-1]
            prediction_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(predictions),
                freq='B'
            )
            
            # Plot predictions
            fig_pred = go.Figure()
            
            # Historical data
            fig_pred.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Predictions
            fig_pred.add_trace(go.Scatter(
                x=prediction_dates,
                y=predictions.flatten(),
                name='Predicted Price',
                line=dict(color='red', dash='dash')
            ))
            
            fig_pred.update_layout(
                title=f'{selected_stock} Stock Price Prediction',
                yaxis_title='Stock Price (USD)',
                template='plotly_dark',
                showlegend=True
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    main()
