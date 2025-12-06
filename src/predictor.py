# src/predictor.py

import pandas as pd
from catboost import CatBoostClassifier
import joblib

class MarketingPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.feature_cols = None  # можно сохранить при обучении

    def predict_proba(self, purchases_df: pd.DataFrame, messages_df: pd.DataFrame):
        # Обработка признаков (можно вынести в отдельный класс)
        X = self.create_features(purchases_df, messages_df)
        
        # Предсказание
        preds = self.model.predict_proba(X)[:, 1]
        return pd.Series(preds, index=X['client_id'], name='target_proba')
    
    def create_features(self, purchases, messages):
        # Копия логики из ноутбука — переносим сюда
        from src.feature_engineering import create_features
        return create_features(purchases, messages)