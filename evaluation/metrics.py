import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    예측 성능 지표 계산

    Parameters:
    - y_true: 실제 값 (shape: [n_samples])
    - y_pred: 예측 값 (shape: [n_samples])

    Returns:
    - dict: RMSE, MAE, MAPE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true + 1e-8)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape
    }


def load_prediction_csv(pred_path: str) -> tuple:
    """
    Autoformer 결과 CSV에서 실제값, 예측값 로딩

    Parameters:
    - pred_path (str): Autoformer output 경로

    Returns:
    - (y_true, y_pred): numpy array tuple
    """
    df = pd.read_csv(pred_path)
    y_true = df["true"].values
    y_pred = df["pred"].values
    return y_true, y_pred

def evaluate_all_etfs(etf_list, output_root="outputs"):
    """
    여러 ETF에 대해 Autoformer 예측 결과 평가

    Parameters:
    - etf_list (list): 예측한 ETF 티커 리스트 (예: ["XLK", "XLF"])
    - output_root (str): 각 ETF 결과가 저장된 폴더의 상위 경로
    """
    all_results = {}

    for etf in etf_list:
        pred_path = os.path.join(output_root, etf, "prediction.csv")
        
        if not os.path.exists(pred_path):
            print(f"[⚠️] {etf}: prediction.csv not found, skipped.")
            continue

        y_true, y_pred = load_prediction_csv(pred_path)
        metrics = evaluate_prediction(y_true, y_pred)

        print(f"[📊] {etf} 평가 지표:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        all_results[etf] = metrics

    return all_results