import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from modeling.baseline_models import LSTMModel, GRUModel, LinearModel

def create_dataloader(X, y, window_size=30, batch_size=32):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_baseline_model(X, y, model_type, etf, output_dir="outputs", window_size=30, epochs=10, device="cpu"):
    input_dim = X.shape[1]

    # 모델 선택
    if model_type == "LSTM":
        model = LSTMModel(input_dim).to(device)
    elif model_type == "GRU":
        model = GRUModel(input_dim).to(device)
    elif model_type == "Linear":
        model = LinearModel(input_dim, window_size).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 데이터 구성
    loader = create_dataloader(X, y, window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 학습
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 예측
    model.eval()
    X_seq = [X[i:i+window_size] for i in range(len(X) - window_size)]
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).squeeze().cpu().numpy()

    # 실제값
    y_true = y[window_size:len(y)]

    # 저장
    os.makedirs(f"{output_dir}/{model_type}", exist_ok=True)
    pred_df = pd.DataFrame({
        "true": y_true,
        "pred": preds
    })
    pred_df.to_csv(f"{output_dir}/{model_type}/{etf}_prediction.csv", index=False)
    print(f"[✔] Saved baseline prediction: {output_dir}/{model_type}/{etf}_prediction.csv")
