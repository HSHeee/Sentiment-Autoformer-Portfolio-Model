# modeling/train_loop.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from modeling.autoformer import Autoformer


def create_encoder_decoder_loader(X, y, enc_len=30, label_len=15, pred_len=5, batch_size=32):
    """
    Autoformer 학습을 위한 Encoder-Decoder 구조 DataLoader 생성

    Parameters:
    - X (np.ndarray): 전체 feature (T, F)
    - y (np.ndarray): 타겟 벡터 (T,)
    - enc_len: encoder 입력 길이
    - label_len: decoder 과거 입력 길이
    - pred_len: decoder 예측 대상 길이

    Returns:
    - DataLoader with (x_enc, x_dec, y_target)
    """
    x_enc_list, x_dec_list, y_target_list = [], [], []

    for i in range(len(X) - enc_len - pred_len):
        x_enc = X[i : i + enc_len]
        x_dec = X[i + enc_len - label_len : i + enc_len + pred_len]
        y_seq = y[i + enc_len : i + enc_len + pred_len]

        x_enc_list.append(x_enc)
        x_dec_list.append(x_dec)
        y_target_list.append(y_seq)

    x_enc_tensor = torch.tensor(x_enc_list, dtype=torch.float32)
    x_dec_tensor = torch.tensor(x_dec_list, dtype=torch.float32)
    y_tensor = torch.tensor(y_target_list, dtype=torch.float32)

    dataset = TensorDataset(x_enc_tensor, x_dec_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model_autoformer(X, y, input_dim, output_dim=1,
                           enc_len=30, label_len=15, pred_len=5,
                           d_model=64, num_enc_layers=2, num_dec_layers=1,
                           lr=1e-3, epochs=10, batch_size=32, device="cpu"):
    """
    Autoformer 학습 루프

    Parameters:
    - X, y: numpy 입력
    - input_dim: feature 수
    - output_dim: 예측할 타겟 수 (보통 1)
    - enc_len, label_len, pred_len: 시계열 길이 설정
    - 기타: 학습 설정

    Returns:
    - 학습된 Autoformer 모델
    """
    loader = create_encoder_decoder_loader(X, y, enc_len, label_len, pred_len, batch_size)
    model = Autoformer(input_dim, d_model, output_dim, num_enc_layers, num_dec_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_enc, x_dec, y_true in loader:
            x_enc, x_dec, y_true = x_enc.to(device), x_dec.to(device), y_true.to(device)

            output = model(x_enc, x_dec)  # (B, pred_len, output_dim)
            output = output.squeeze(-1)  # (B, pred_len)

            loss = criterion(output, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return model
