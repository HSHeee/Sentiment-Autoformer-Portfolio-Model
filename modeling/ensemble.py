import numpy as np
import pandas as pd

def ensemble_predictions(etf, models, output_dir="outputs"):
    predictions = []
    for model in models:
        pred_path = f"{output_dir}/{model}/{etf}_prediction.csv"
        df = pd.read_csv(pred_path)
        predictions.append(df["pred"].values)

    # 평균 앙상블
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred