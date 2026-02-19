import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("📈 Stokify AI - Simple Stock Predictor")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")

if stock:
    data = yf.download(stock, start="2022-01-01", end="2024-01-01")

    if not data.empty:
        st.subheader("Stock Data")
        st.write(data.tail())

        data['MA50'] = data['Close'].rolling(50).mean()

        st.subheader("Moving Average Chart")
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close Price')
        ax.plot(data['MA50'], label='MA50')
        ax.legend()
        st.pyplot(fig)

        data = data.dropna()
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values

        model = LinearRegression()
        model.fit(X, y)

        future_day = np.array([[len(data) + 1]])
        prediction = model.predict(future_day)

        predicted_price = round(prediction[0][0], 2)

        st.subheader("📈 Predicted Next Day Price")
        st.success(f"Predicted Price: ${predicted_price}")

    else:
        st.write("Invalid stock symbol")
