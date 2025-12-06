# src/feature_engineering.py

import pandas as pd
import ast

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def create_features(purchases, messages):
    # Обработка категорий
    purchases['categories_list'] = purchases['category_ids'].apply(safe_eval)
    
    # Топ-50 категорий (можно сохранить как константу или обучать в fit)
    all_cats = purchases.explode('categories_list')['categories_list'].value_counts().head(50).index.tolist()
    top_cats = set(all_cats)
    
    for cat in top_cats:
        col_name = f'cat_{cat}'
        purchases[col_name] = purchases['categories_list'].apply(lambda x: 1 if cat in x else 0)
    
    # Агрегация
    client_purch = purchases.groupby('client_id').agg(
        total_quantity=('quantity', 'sum'),
        avg_price=('price', 'mean'),
        n_purchases=('client_id', 'count'),
        n_unique_categories=('categories_list', lambda x: len(set('|'.join(map(str, x)).split('|'))))
    ).reset_index()
    
    client_msgs = messages.groupby('client_id').agg(
        total_messages=('message_id', 'count'),
        opened_rate=('event', lambda x: (x == 'opened').mean()),
        purchased_from_msg=('event', lambda x: (x == 'purchased').sum()),
        n_unique_campaigns=('bulk_campaign_id', 'nunique')
    ).reset_index()
    
    X = client_purch.merge(client_msgs, on='client_id', how='outer')
    return X.fillna(0)