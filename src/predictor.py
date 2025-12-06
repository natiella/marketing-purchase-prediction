from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        early_stopping_rounds=100,
        verbose=False
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    scores.append(roc_auc_score(y_val, preds))

print(f"ROC AUC на CV: {np.mean(scores):.4f}")