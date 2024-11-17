import pandas as pd
import numpy as np
from datetime import datetime
from pycoingecko import CoinGeckoAPI
import ccxt
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Initialize API clients
cg = CoinGeckoAPI()
binance = ccxt.binance()

# Function to fetch additional DOGE/USDT data from OKX API
def fetch_doge_data_okx():
    url = "https://www.okx.com/api/v5/market/ticker?instId=DOGE-USDT"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for unsuccessful responses
        data = response.json()['data'][0]  # Extract the first item in 'data' list

        # Extract various data points
        price = float(data['last'])
        open_price = float(data['open24h'])
        high_price = float(data['high24h'])
        low_price = float(data['low24h'])
        volume_24h = float(data['volCcy24h'])  # Volume in quote currency (USDT)
        bid_price = float(data['bidPx'])
        ask_price = float(data['askPx'])
        percentage_change_24h = float(data['sodUtc0'])

        return {
            'last_price': price,
            'open_price_24h': open_price,
            'high_price_24h': high_price,
            'low_price_24h': low_price,
            'volume_24h': volume_24h,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'percentage_change_24h': percentage_change_24h
        }

    except requests.RequestException as e:
        print("Error fetching data from OKX:", e)
        return None

# Function to get historical data from CoinGecko
def get_historical_data_from_coingecko(days=180):
    try:
        data = cg.get_coin_market_chart_by_id(id='dogecoin', vs_currency='usd', days=days)
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print("Failed to get data from CoinGecko:", e)
        return None

# Combined function to get data from available sources
def get_historical_data(days=180):
    df = get_historical_data_from_coingecko(days)
    if df is not None and len(df) >= 24:
        print("Data fetched from CoinGecko.")
        return df

    print("Insufficient data from all sources.")
    return None

# Feature engineering for LSTM, including additional OKX data
def create_features(df, okx_data):
    df['returns'] = df['price'].pct_change(fill_method=None)
    df['MA5'] = df['price'].rolling(window=5).mean()
    df['MA15'] = df['price'].rolling(window=15).mean()
    df['Bollinger_Upper'] = df['price'].rolling(20).mean() + df['price'].rolling(20).std() * 2
    df['Bollinger_Lower'] = df['price'].rolling(20).mean() - df['price'].rolling(20).std() * 2
    df['RSI'] = compute_rsi(df['price'])

    # Adding OKX data as constant features for the last row
    df['high_price_24h'] = okx_data['high_price_24h']
    df['low_price_24h'] = okx_data['low_price_24h']
    df['volume_24h'] = okx_data['volume_24h']
    df['bid_price'] = okx_data['bid_price']
    df['ask_price'] = okx_data['ask_price']
    df['percentage_change_24h'] = okx_data['percentage_change_24h']
    
    df.dropna(inplace=True)
    return df

# Calculate RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Prepare data for LSTM
def prepare_data(df, sequence_length=24):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['price', 'returns', 'MA5', 'MA15', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'high_price_24h', 'low_price_24h', 'volume_24h', 'bid_price', 'ask_price', 'percentage_change_24h']])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(1 if scaled_data[i, 0] > scaled_data[i-1, 0] else 0)  # 1 for up, 0 for down

    return np.array(X), np.array(y), scaler

# Build improved model with GRU and additional layers
def build_improved_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(df):
    X, y, scaler = prepare_data(df)
    model = build_improved_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=15, batch_size=32, verbose=1)
    return model, scaler

# Prediction function for the next 24 hours
def predict_24h(model, scaler, df, sequence_length=24):
    if len(df) < sequence_length:
        raise ValueError(f"Not enough data to make a 24-hour prediction. Expected at least {sequence_length} rows, got {len(df)} rows.")
    
    X_live = df[['price', 'returns', 'MA5', 'MA15', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'high_price_24h', 'low_price_24h', 'volume_24h', 'bid_price', 'ask_price', 'percentage_change_24h']].values[-sequence_length:]
    X_live_scaled = scaler.transform(X_live)
    X_live_scaled = np.expand_dims(X_live_scaled, axis=0)

    pred = model.predict(X_live_scaled)[0][0]
    direction = "up" if pred > 0.5 else "down"
    buy_confidence = round(pred * 100, 2) if direction == "up" else round((1 - pred) * 100, 2)
    sell_confidence = 100 - buy_confidence

    return {
        'direction': direction,
        'buy_confidence': buy_confidence,
        'sell_confidence': sell_confidence
    }

# Main function
def main():
    okx_data = fetch_doge_data_okx()
    if okx_data is None:
        print("Failed to fetch OKX data.")
        return

    data = get_historical_data(days=180)
    if data is None or len(data) < 24:
        print("Not enough data available from any source to make predictions.")
        return

    # Resample data to hourly intervals for 24-hour predictions
    data['price'] = data['price'].resample('1h').last().ffill()
    data = create_features(data, okx_data)
    
    if len(data) < 24:
        print("Not enough data for 24-hour interval predictions after resampling. Please provide more historical data.")
        return
    
    model, scaler = train_model(data)
    prediction = predict_24h(model, scaler, data)
    print(f"24-hour prediction: {prediction['direction']} | Buy Confidence: {prediction['buy_confidence']}% | Sell Confidence: {prediction['sell_confidence']}%")

# Run the script
if __name__ == "__main__":
    main()
