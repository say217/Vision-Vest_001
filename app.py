import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler   
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
import random
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import seaborn as sns
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# Flask app setup
app = Flask(__name__)

# Reproducibility helpers
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
class Config:
    def __init__(self, ticker_symbol: str):
        self.ticker = ticker_symbol.upper()
        self.start_date = "2022-01-01"
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.features = ['Close', 'High', 'Low']
        self.technical_indicators = [
            'SMA_10', 'RSI', 'MACD',
            'BB_Upper', 'BB_Lower', 'ATR',
            'Price_Change', 'Log_Close', 'Volatility_10',
            'Momentum_5', 'Momentum_10', 'EMA_10', 'EMA_20', 'Volatility_20', 'Volatility_50'
        ]
        self.window_size = 40
        self.train_split = 0.8
        self.forecast_days = 4
        self.past_days_plot = 14
        self.batch_size = 64
        self.epochs = 70
        self.learning_rate = 5e-4
        self.lstm_units = 192
        self.num_layers = 3
        self.dropout = 0.25
        self.loss = 'huber'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_features = None

# Data handling
class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = MinMaxScaler()
        self.dates = None
        self.n_features = None
        self.output_log = []

    def log(self, message):
        self.output_log.append(message)

    def download_data(self):
        cache_file = f'data_{self.config.ticker}.pkl'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.dates = data.index
                self.log(f"Loaded cached data for {self.config.ticker}")
                return data
            except Exception:
                self.log(f"Cache file corrupted, re-downloading data for {self.config.ticker}")

        try:
            self.log(f"Downloading data for {self.config.ticker}...")
            data = yf.download(
                self.config.ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d",
                progress=False
            )
            if data.empty:
                self.log(f"No data retrieved for ticker {self.config.ticker}")
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
            available_cols = [col for col in required_cols if col in data.columns]
            if 'Close' not in available_cols:
                self.log(f"Critical error: 'Close' price not available for {self.config.ticker}")
                return None
            self.config.features = [col for col in self.config.features if col in available_cols]
            if len(data) < self.config.window_size:
                self.log(f"Insufficient data: {len(data)} samples, need at least {self.config.window_size}")
                return None
            data = data[self.config.features].copy()
            data = self._add_technical_indicators(data)
            data = self._handle_missing_values(data)
            self.dates = data.index
            self.log(f"Data shape after preprocessing: {data.shape}")
            self.log(f"Date range: {data.index[0]} to {data.index[-1]}")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return data
        except Exception as e:
            self.log(f"Error downloading data for {self.config.ticker}: {e}")
            return None

    def _add_technical_indicators(self, data: pd.DataFrame):
        try:
            data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
            data['RSI'] = self._compute_rsi(data['Close'])
            data['MACD'] = self._compute_macd(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = self._compute_bollinger_bands(data['Close'])
            data['ATR'] = self._compute_atr(data)
            data['Price_Change'] = data['Close'].pct_change()
            data['Log_Close'] = np.log(data['Close'] + 1e-8)
            data['Volatility_10'] = data['Close'].pct_change().rolling(10, min_periods=1).std()
            data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
            data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
            data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            data['Volatility_20'] = data['Close'].pct_change().rolling(20, min_periods=1).std()
            data['Volatility_50'] = data['Close'].pct_change().rolling(50, min_periods=1).std()
        except Exception as e:
            self.log(f"Error computing technical indicators: {e}")
            for col in self.config.technical_indicators:
                if col not in data.columns:
                    data[col] = data['Close']
        return data

    def _handle_missing_values(self, data: pd.DataFrame):
        data = data.fillna(method='ffill')
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())
        data = data.fillna(0)
        return data

    def _compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _compute_macd(self, prices, slow=26, fast=12, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def _compute_bollinger_bands(self, prices, window=20, num_std=2):
        sma = prices.rolling(window=window, min_periods=1).mean()
        std = prices.rolling(window=window, min_periods=1).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper.fillna(sma), lower.fillna(sma)

    def _compute_atr(self, data, period=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr.fillna(atr.mean())

    def prepare_data(self, data: pd.DataFrame):
        feature_list = list(dict.fromkeys(self.config.features + self.config.technical_indicators))
        available_features = [f for f in feature_list if f in data.columns]
        self.log(f"Using features: {available_features}")
        self.n_features = len(available_features)
        features_vals = data[available_features].values
        scaled_features = self.scaler.fit_transform(features_vals)
        X, y = [], []
        W = self.config.window_size
        for i in range(len(scaled_features) - W):
            X.append(scaled_features[i:i + W])
            y.append(scaled_features[i + W, available_features.index('Close')])
        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * self.config.train_split)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        self.log(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        return (X_train, y_train), (X_test, y_test), scaled_features

    def inverse_target_transform(self, scaled_data):
        scaled = np.asarray(scaled_data).reshape(-1, 1)
        dummy = np.zeros((len(scaled), self.n_features))
        dummy[:, 0] = scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]

# Dataset
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        score = self.tanh(self.attn(lstm_out))
        attn_weights = self.softmax(torch.matmul(score, self.v))
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out)
        return context.squeeze(1), attn_weights

# Enhanced Model (BiGRU + Attention)
class EnhancedGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.n_features
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_size, input_size, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.gru = nn.GRU(
            input_size=input_size*2,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.15)
        self.ln = nn.LayerNorm(256)

    def forward(self, x):
        conv_out1 = self.act(self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1))
        conv_out2 = self.act(self.conv2(x.permute(0, 2, 1)).permute(0, 2, 1))
        conv_out = torch.cat([conv_out1, conv_out2], dim=2)
        gru_out, _ = self.gru(conv_out)
        scores = torch.matmul(self.attn(gru_out), gru_out.transpose(1,2)) / (gru_out.size(-1)**0.5)
        weights = torch.softmax(scores.mean(dim=1), dim=1).unsqueeze(-1)
        context = (weights * gru_out).sum(dim=1)
        context = self.ln(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out.squeeze(-1)

# Trainer
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.output_log = []

    def log(self, message):
        self.output_log.append(message)

    def train(self, model, train_loader, test_loader):
        if self.config.loss == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        scaler = GradScaler() if torch.cuda.is_available() else None
        best_loss = float('inf')
        train_losses, test_losses = [], []
        patience, early_stopping = 15, False
        patience_counter = 0
        start_time = time.time()
        os.makedirs("models", exist_ok=True)
        for epoch in range(self.config.epochs):
            if early_stopping:
                break
            model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                if scaler:
                    with autocast():
                        output = model(X)
                        loss = criterion(output, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                train_loss += loss.item() * X.size(0)
            train_loss /= len(train_loader.dataset)
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    if scaler:
                        with autocast():
                            output = model(X)
                            batch_loss = criterion(output, y).item()
                    else:
                        output = model(X)
                        batch_loss = criterion(output, y).item()
                    test_loss += batch_loss * X.size(0)
            test_loss /= len(test_loader.dataset)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            scheduler.step(test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), f'models/{self.config.ticker}_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log("Early stopping triggered")
                    early_stopping = True
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.log(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
                )
        self.log(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
        return train_losses, test_losses

    def evaluate(self, model, loader, data_handler: DataHandler):
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                output = model(X).cpu().numpy()
                preds.extend(data_handler.inverse_target_transform(output))
                actuals.extend(data_handler.inverse_target_transform(y.cpu().numpy()))
        return np.array(preds), np.array(actuals)

    def directional_accuracy(self, actuals, preds):
        if len(actuals) <= 1 or len(preds) <= 1:
            return 0.0
        actual_diff = np.diff(actuals)
        pred_diff = np.diff(preds)
        correct = np.sum((actual_diff > 0) == (pred_diff > 0))
        return correct / len(actual_diff) * 100.0

# Visualization functions
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1c2526', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

def plot_results(dates, actuals, preds, title, config):
    fig = plt.figure(figsize=(12, 6), facecolor='#1c2526')
    plt.plot(dates, actuals, label='Actual', color='#00ff00', linewidth=2)
    plt.plot(dates, preds, '--', label='Predicted', color='#00f7ff', linewidth=2)
    plt.title(f"{title} - {config.ticker}", fontsize=14, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price (USD)', fontsize=12, color='white')
    plt.legend(fontsize=10, facecolor='#1c2526', edgecolor='white', labelcolor='white')
    plt.grid(True, alpha=0.3, color='#cccccc')
    plt.gcf().autofmt_xdate()
    plt.tick_params(colors='white')
    plt.tight_layout()
    return plot_to_base64(fig)

def plot_forecast(dates, prices, forecast_dates, forecast_prices, std, config):
    fig = plt.figure(figsize=(14, 8), facecolor='#1c2526')
    past_days = config.past_days_plot
    historical_dates = dates[-past_days:]
    historical_prices = prices[-past_days:]
    plt.plot(historical_dates, historical_prices, 'o-', label='Historical (Past 60 Days)', color='#ff00ff', linewidth=2)
    plt.plot(forecast_dates, forecast_prices, 'o-', label=f'Forecast (Next {config.forecast_days} Days)', color='#ffff00', linewidth=2, markersize=6)
    plt.fill_between(forecast_dates, forecast_prices - std, forecast_prices + std, alpha=0.2, color='#ffff00', label='Confidence Interval')
    plt.axvline(x=dates[-1], color='#cccccc', linestyle='--', alpha=0.7, label='Today')
    plt.title(f"{config.ticker} - Past {past_days} Days & {config.forecast_days}-Day Forecast", fontsize=14, color='white')
    plt.xlabel('Date', fontsize=12, color='white')
    plt.ylabel('Price (USD)', fontsize=12, color='white')
    plt.legend(fontsize=10, facecolor='#1c2526', edgecolor='white', labelcolor='white')
    plt.grid(True, alpha=0.3, color='#cccccc')
    plt.gcf().autofmt_xdate()
    plt.tick_params(colors='white')
    plt.tight_layout()
    return plot_to_base64(fig)

def plot_frequency_and_heatmap(data, ticker, features, technical_indicators):
    plt.style.use('dark_background')
    output_log = []
    all_features = list(dict.fromkeys(features + technical_indicators))
    available_features = [f for f in all_features if f in data.columns]
    hist_features = ['Close', 'RSI', 'SMA_10', 'BB_Upper', 'BB_Lower', 'Volatility_10', 'Volatility_20', 'Price_Change', 'MACD']
    hist_features = [f for f in hist_features if f in available_features]
    plot_images = []
    if hist_features:
        fig = plt.figure(figsize=(15, 10), facecolor='#1c2526')
        colors = ['#00ffab', '#ff6f61', '#ffd700', '#6ab04c', '#ff85ff', '#00b7eb', '#ff9f43', '#5c5c8a', '#ff4f81']
        for i, feature in enumerate(hist_features, 1):
            plt.subplot(3, 3, i)
            data_clean = data[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if data_clean.empty:
                output_log.append(f"Warning: No valid data for {feature}. Skipping histogram.")
                continue
            sns.histplot(data_clean, bins=20, kde=True, color=colors[i-1], edgecolor='white', alpha=0.7)
            plt.title(f'{feature} Distribution', fontsize=10, color='white')
            plt.xlabel(feature, fontsize=8, color='white')
            plt.ylabel('Frequency', fontsize=8, color='white')
            plt.grid(True, alpha=0.3, color='gray')
            plt.tick_params(colors='white')
        plt.suptitle(f'{ticker} Feature Distributions', fontsize=14, color='white')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_images.append(plot_to_base64(fig))
    else:
        output_log.append("Warning: No valid features for histogram. Skipping frequency plot.")
    fig = plt.figure(figsize=(12, 10), facecolor='#1c2526')
    correlation_matrix = data[available_features].corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='Spectral',
        center=0,
        vmin=-1,
        vmax=1,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        annot_kws={'size': 8, 'color': 'white'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'{ticker} Feature Correlation Heatmap', fontsize=14, color='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    plt.tight_layout()
    plot_images.append(plot_to_base64(fig))
    return plot_images, output_log

def predict_future(model, last_window, num_days, data_handler, config):
    model.eval()
    predictions_scaled = []
    current_window = last_window.copy()
    for _ in range(num_days):
        input_tensor = torch.FloatTensor(current_window).unsqueeze(0).to(config.device)
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().item()
        predictions_scaled.append(pred_scaled)
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1, 0] = pred_scaled
    return data_handler.inverse_target_transform(np.array(predictions_scaled))

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker', 'AAPL').upper()
    config = Config(ticker)
    os.makedirs('models', exist_ok=True)
    data_handler = DataHandler(config)
    data = data_handler.download_data()
    if data is None or data.empty:
        return jsonify({
            'error': f"Failed to download data for {ticker}. Please check the ticker symbol.",
            'output_log': data_handler.output_log
        })
    if data_handler.dates is None:
        return jsonify({
            'error': f"Date index not set for {ticker}. Data download failed.",
            'output_log': data_handler.output_log
        })
    # Generate visualizations
    plot_images, vis_log = plot_frequency_and_heatmap(data, ticker, config.features, config.technical_indicators)
    data_handler.output_log.extend(vis_log)
    # Prepare data
    (X_train, y_train), (X_test, y_test), scaled_data = data_handler.prepare_data(data)
    config.n_features = data_handler.n_features
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    data_handler.log(f"Initializing model on device: {config.device}")
    model = EnhancedGRU(config).to(config.device)
    trainer = Trainer(config)
    data_handler.log("Starting model training...")
    train_losses, test_losses = trainer.train(model, train_loader, test_loader)
    data_handler.output_log.extend(trainer.output_log)
    # Load best checkpoint
    try:
        model.load_state_dict(torch.load(f'models/{ticker}_model.pth', map_location=config.device))
        data_handler.log(f"Loaded best model for {ticker}")
    except FileNotFoundError:
        data_handler.log("Warning: Model file not found. Using current model state.")
    # Evaluate
    test_preds, test_actuals = trainer.evaluate(model, test_loader, data_handler)
    performance_metrics = {}
    if len(test_actuals) > 0 and len(test_preds) > 0:
        rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
        mae = mean_absolute_error(test_actuals, test_preds)
        r2 = r2_score(test_actuals, test_preds) * 100
        directional_acc = trainer.directional_accuracy(test_actuals, test_preds)
        performance_metrics = {
            'rmse': f"{rmse:.2f}",
            'mae': f"${mae:.2f}",
            'r2': f"{r2:.1f}",
            'directional_acc': f"{directional_acc:.2f}%"
        }
        data_handler.log(f"\n{ticker} Model Performance:")
        data_handler.log("-" * 40)
        data_handler.log(f"RMSE: ${rmse:.2f}")
        data_handler.log(f"MAE: ${mae:.2f}")
        data_handler.log(f"R² Score: {r2:.1f}")
        data_handler.log(f"Directional Accuracy: {directional_acc:.2f}%")
        test_dates = data_handler.dates[len(X_train) + config.window_size:]
        if len(test_dates) >= len(test_actuals):
            test_plot = plot_results(test_dates[:len(test_actuals)], test_actuals, test_preds, f"Test Predictions", config)
        else:
            test_plot = None
    else:
        test_plot = None
    # Forecast
    data_handler.log(f"\nGenerating {config.forecast_days}-day forecast...")
    last_window = scaled_data[-config.window_size:]
    future_prices = predict_future(model, last_window, config.forecast_days, data_handler, config)
    start_date = datetime.now() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(config.forecast_days)]
    forecast_table = [
        {'date': date.strftime('%Y-%m-%d (%A)'), 'price': f"${price:.2f}"}
        for date, price in zip(future_dates, future_prices)
    ]
    data_handler.log(f"\n{ticker} - {config.forecast_days} Day Forecast:")
    data_handler.log("-" * 50)
    for item in forecast_table:
        data_handler.log(f"{item['date']}: {item['price']}")
    if len(test_preds) > 0 and len(test_actuals) > 0:
        std = np.std(test_preds - test_actuals)
    else:
        std = np.std(future_prices) * 0.1
    historical_prices = data_handler.inverse_target_transform(scaled_data[:, 0])
    forecast_plot = plot_forecast(data_handler.dates, historical_prices, future_dates, future_prices, std, config)
    data_handler.log(f"\nAnalysis completed for {ticker}")
    return jsonify({
        'output_log': data_handler.output_log,
        'performance_metrics': performance_metrics,
        'forecast_table': forecast_table,
        'plot_images': plot_images,
        'test_plot': test_plot,
        'forecast_plot': forecast_plot
    })
    
    if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
