import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("fivethirtyeight")

app = Flask(
    __name__,
    template_folder=os.path.join(os.getcwd(), "templates"),
    static_folder=os.path.join(os.getcwd(), "static")
)

# Load trained model
model = load_model('stock_dl_model.h5')

# List of supported stocks (Dropdown)
stock_list = [
    "POWERGRID.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
    "BHARTIARTL.NS"
]

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        stock = request.form.get('stock')

        # Safety check
        if stock not in stock_list:
            stock = "POWERGRID.NS"

        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 10, 1)

        # API Call Initialization
        print(f"[INFO] Fetching data for: {stock}")
        print(f"[INFO] Source: Yahoo Finance API via yfinance")

        # Download stock data
        df = yf.download(stock, start=start, end=end)

        # Success message
        if not df.empty:
            print(f"[SUCCESS] Data downloaded successfully for {stock}")
            print(f"[INFO] Total records fetched: {len(df)}")
        else:
            print(f"[ERROR] No data found for {stock}")
            return render_template('index.html', stock_list=stock_list, error="No data found!")

        data_desc = df.describe()

        # EMAs
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Train/Test Split
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Prediction
        y_predicted = model.predict(x_test)

        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot 1
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, label='Closing Price')
        ax1.plot(ema20, label='EMA 20')
        ax1.plot(ema50, label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 EMA)")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, label='Closing Price')
        ax2.plot(ema100, label='EMA 100')
        ax2.plot(ema200, label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 EMA)")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, label="Original Price")
        ax3.plot(y_predicted, label="Predicted Price")
        ax3.set_title("Prediction vs Original Trend")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        return render_template(
            'index.html',
            stock_list=stock_list,
            selected_stock=stock,
            plot_path_ema_20_50=ema_chart_path,
            plot_path_ema_100_200=ema_chart_path_100_200,
            plot_path_prediction=prediction_chart_path,
            data_desc=data_desc.to_html(classes='table table-bordered'),
            dataset_link=csv_file_path
        )

    return render_template('index.html', stock_list=stock_list)


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
