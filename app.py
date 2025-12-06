# app.py

from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
import sys, os
sys.path.append('src')
from src.predictor import MarketingPredictor

app = FastAPI()

predictor = MarketingPredictor('models/marketing_model.pkl')

@app.post("/predict")
async def predict(
    purchases: UploadFile = File(...),
    messages: UploadFile = File(...)
):
    df_p = pd.read_csv(io.BytesIO(await purchases.read()))
    df_m = pd.read_csv(io.BytesIO(await messages.read()))
    preds = predictor.predict_proba(df_p, df_m)
    return preds.reset_index().to_dict(orient='records')
