import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class ReturnPredictor:
    """
    Machine Learning model to predict future asset returns.
    """

    def __init__(self, model_type='rf', n_estimators=100, max_depth=None):
        """
        Args:
            model_type (str): 'rf' for Random Forest (can be extended).
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Max depth of the tree.
        """
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            raise ValueError("Only 'rf' (Random Forest) is currently supported.")
        
        self.feature_cols = []

    def _engineer_features(self, prices):
        """
        Creates technical indicators as features.
        
        Args:
            prices (pd.Series): Asset price series.
        
        Returns:
            pd.DataFrame: DataFrame with features and target (shifted return).
        """
        df = prices.to_frame(name='Close')
        
        # Returns
        df['Return'] = df['Close'].pct_change()
        
        # 1. Momentum (Lagged Returns)
        for lag in [1, 3, 5, 21]:
            df[f'Ret_Lag_{lag}'] = df['Return'].shift(lag)
            
        # 2. Rolling Volatility
        for window in [21, 63]:
            df[f'Vol_{window}'] = df['Return'].rolling(window).std()
            
        # 3. Simple Moving Averages (Trend)
        df['SMA_21'] = df['Close'].rolling(21).mean() / df['Close'] - 1 # Normalized to price
        
        # Target: Next day return
        df['Target'] = df['Return'].shift(-1)
        
        df_dropped = df.dropna()
        if df_dropped.empty:
            print(f"Warning: Dropped all data in feature engineering. Original len: {len(df)}")
            
        return df_dropped

    def fit(self, prices_df):
        """
        Trains the model on price data for multiple assets.
        
        Args:
            prices_df (pd.DataFrame): DataFrame with columns as tickers and rows as dates (prices).
        """
        X_all = []
        y_all = []
        
        for ticker in prices_df.columns:
            features_df = self._engineer_features(prices_df[ticker])
            
            # Identify feature columns (exclude Target and raw Close/Return if desired, but engineered ones are key)
            self.feature_cols = [c for c in features_df.columns if c not in ['Close', 'Return', 'Target']]
            
            X_all.append(features_df[self.feature_cols].values)
            y_all.append(features_df['Target'].values)
            
        # Concatenate all asset data for a global model (or could train per asset)
        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        
        # Train
        self.model.fit(X, y)
        print("Model trained.")

    def predict(self, current_prices_df):
        """
        Predicts next period return for each asset based on recent data.
        
        Args:
            current_prices_df (pd.DataFrame): DataFrame with sufficient history to generate features.
        
        Returns:
            pd.Series: Predicted returns for each asset.
        """
        predictions = {}
        
        for ticker in current_prices_df.columns:
            features_df = self._engineer_features(current_prices_df[ticker])
            
            if features_df.empty:
                print(f"Not enough data for {ticker}")
                predictions[ticker] = np.nan
                continue
                
            # Take the last row (most recent known state)
            # Note: _engineer_features drops NA, so the last row has valid features derived from recent history.
            # We want to predict for T+1, so we use features at T.
            last_features = features_df.iloc[[-1]][self.feature_cols]
            
            pred = self.model.predict(last_features.values)
            predictions[ticker] = pred[0]
            
        return pd.Series(predictions)

    def evaluate(self, prices_df, test_size=0.2):
        """
        Evaluates the model using a simple train/test split on the provided data.
        """
        X_all = []
        y_all = []
        
        for ticker in prices_df.columns:
            features_df = self._engineer_features(prices_df[ticker])
            self.feature_cols = [c for c in features_df.columns if c not in ['Close', 'Return', 'Target']]
            X_all.append(features_df[self.feature_cols].values)
            y_all.append(features_df['Target'].values)
            
        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        return {'MSE': mse, 'R2': r2}
