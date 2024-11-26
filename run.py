import ccxt
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class SwingTradingBotSpot:
    def __init__(self, trading_pair, interval, capital, risk_per_trade=0.02, limit=500):
        self.trading_pair = trading_pair
        self.interval = interval
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.limit = limit
        self.exchange = ccxt.okx()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def fetch_historical_data(self):
        """Fetch historical OHLCV data using CCXT."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.trading_pair, timeframe=self.interval, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        """Calculate technical indicators."""
        try:
            df["RSI"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"])
            if not macd.empty:
                df["MACD"], df["MACD_signal"] = macd["MACD_12_26_9"], macd["MACDs_12_26_9"]
            else:
                df["MACD"], df["MACD_signal"] = None, None
            df["EMA_9"] = ta.ema(df["close"], length=9)
            df["EMA_21"] = ta.ema(df["close"], length=21)
            bollinger = ta.bbands(df["close"], length=20)
            if not bollinger.empty:
                df["Upper_Band"], df["Middle_Band"], df["Lower_Band"] = (
                    bollinger["BBU_20_2.0"], bollinger["BBM_20_2.0"], bollinger["BBL_20_2.0"]
                )
            df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["Price_Change"] = df["close"].pct_change()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df

    def prepare_data_for_model(self, df):
        """Prepare data for the TensorFlow model."""
        df["Target"] = (df["Price_Change"].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)
        features = df[[
            "RSI", "MACD", "MACD_signal", "EMA_9", "EMA_21",
            "Upper_Band", "Middle_Band", "Lower_Band", "ATR", "Price_Change"
        ]]
        target = df["Target"]
        scaled_features = self.scaler.fit_transform(features)
        return np.array(scaled_features), np.array(target)

    def build_and_train_model(self, X_train, y_train):
        """Build and train a TensorFlow model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        self.model = model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and return accuracy."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return accuracy * 100

    def predict_signal(self, latest_data):
        """Predict the trading signal and calculate percentage change."""
        scaled_data = self.scaler.transform([latest_data])
        prediction = self.model.predict(scaled_data)[0][0]

        if prediction > 0.6:
            action = "Buy"
            percentage_change = (prediction - 0.5) * 2 * 100  # Scale to percentage above 50%
            percentage_change_str = f"+{percentage_change:.2f}%"  # Ensure + for positive
        elif prediction < 0.4:
            action = "Sell"
            percentage_change = (0.5 - prediction) * 2 * 100  # Scale to percentage below 50%
            percentage_change_str = f"-{percentage_change:.2f}%"  # Ensure - for negative
        else:
            action = "Hold"
            percentage_change_str = "0.00%"  # No movement predicted

        return action, percentage_change_str

    def run(self):
        """Run the swing trading bot."""
        print(f"Fetching historical data for {self.trading_pair}...")
        data = self.fetch_historical_data()
        if data.empty:
            print("Failed to fetch data.")
            return

        print("Calculating indicators...")
        data = self.calculate_indicators(data)
        if data.empty:
            print("Failed to calculate indicators.")
            return

        print("Preparing data for model...")
        X, y = self.prepare_data_for_model(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Building and training the TensorFlow model...")
        self.build_and_train_model(X_train, y_train)

        print("Evaluating the model...")
        accuracy = self.evaluate_model(X_test, y_test)
        print(f"Model Accuracy: {accuracy:.2f}%")

        print("Generating trading signals...")
        latest_row = data.iloc[-1][[
            "RSI", "MACD", "MACD_signal", "EMA_9", "EMA_21",
            "Upper_Band", "Middle_Band", "Lower_Band", "ATR", "Price_Change"
        ]].values
        action, percentage_change_str = self.predict_signal(latest_row)

        print(f"Trading Recommendation for {self.trading_pair} on {self.interval} interval:")
        print(f"You need to {action}")
        print(f"Predicted Price Movement in 24 hours: {percentage_change_str}")

# Main Execution
if __name__ == "__main__":
    bot = SwingTradingBotSpot("DOGE/USDT", "1d", capital=500)
    bot.run()
