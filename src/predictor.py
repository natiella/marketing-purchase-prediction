# src/predictor.py

import pandas as pd
from .feature_engineering import create_features
from catboost import CatBoostClassifier
import joblib

class MarketingPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.feature_cols = None  # можно сохранить при обучении

    def predict_proba(self, purchases_df: pd.DataFrame, messages_df: pd.DataFrame):
        X = create_features(purchases_df, messages_df)
        # Убедитесь, что колонки совпадают с обучением
        preds = self.model.predict_proba(X[self.model.feature_names_])[:, 1]
        return pd.Series(preds, index=X['client_id'], name='target_proba')