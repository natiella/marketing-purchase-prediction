# Прогноз покупки в течение 90 дней

## Метрика
- **ROC AUC** (обоснование: задача бинарной классификации с дисбалансом)

## Структура
- `notebooks/research.ipynb` — исследование, feature engineering, обучение
- `src/` — `FeatureEngineering`, `MarketingPredictor`
- `submission.csv` — `client_id`, `target_proba`

## Зависимости
```bash
pip install -r requirements.txt