# predict.py

import pandas as pd
import sys, os
sys.path.append('src')
from src.predictor import MarketingPredictor

def main():
    # Загружаем данные
    purchases = pd.read_csv('data/apparel-purchases.csv')
    messages = pd.read_csv('data/apparel-messages.csv')
    
    # Предсказание
    predictor = MarketingPredictor('models/marketing_model.pkl')
    result = predictor.predict_proba(purchases, messages)
    
    # Сохраняем
    result.reset_index().to_csv('submission.csv', index=False)
    print("✅ submission.csv сохранён")

if __name__ == "__main__":
    main()
    