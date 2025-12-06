# predict.py

import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.predictor import MarketingPredictor

def main():
    # Загрузка данных
    purchases = pd.read_csv('data/apparel-purchases.csv')
    messages = pd.read_csv('data/apparel-messages.csv')
    
    # Предсказание
    predictor = MarketingPredictor(
        model_path='models/marketing_model.pkl',
        preprocessor_path='models/preprocessor.pkl'  # если есть
    )
    result = predictor.predict_proba(purchases, messages)
    
    # Сохранение
    result.reset_index().to_csv('submission.csv', index=False)
    print(f"✅ Готово! Результат сохранён в submission.csv")

if __name__ == "__main__":
    main()