import sys
sys.path.append("C:\Project\Autoformer")
import os
import subprocess
import pandas as pd

def train_autoformer(
    etf: str,
    input_dir: str = "data/autoformer_input",
    output_dir: str = "outputs",
    pred_len: int = 5,
    target: str = "close"
):
    """
    공식 Autoformer run.py를 subprocess로 실행해 학습 시작

    Parameters:
    - etf (str): 예측할 ETF 이름 (ex: 'XLK')
    - input_dir (str): Autoformer용 CSV가 저장된 경로
    - output_dir (str): 결과가 저장될 폴더
    - pred_len (int): 예측 길이 (예: 1일, 5일 등)
    - target (str): 예측 대상 열
    """
    root_path = input_dir
    data_path = f"{etf}_autoformer.csv"

    # 입력 feature 개수 자동 계산
    csv_path = os.path.join(input_dir, data_path)
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col not in ["date", target]]
    enc_in = len(feature_cols) + 1  # +1은 target 포함(Multivariate 예측시)
    dec_in = enc_in
    c_out = 1

    # command-line 인자 구성
    command = [
        "python", "C:\Project\Autoformer/run.py",
        "--is_training", "1",
        "--root_path", root_path,
        "--data_path", data_path,
        "--model_id", f"{etf}_{pred_len}d",
        "--model", "Autoformer",
        "--data", "custom",
        "--features", "M",             # Multivariate
        "--target", target,
        "--seq_len", "60",
        "--label_len", "30",
        "--pred_len", str(pred_len),
        "--e_layers", "2",
        "--d_layers", "1",
        "--factor", "3",
        "--enc_in", str(enc_in),             # placeholder, Autoformer 내부에서 계산
        "--dec_in", str(dec_in),
        "--c_out", str(c_out),
        "--des", "Exp",
        "--itr", "1",
        "--train_epochs", "5",
        "--batch_size", "32",
        "--learning_rate", "0.001",
        "--patience", "2",
        "--checkpoints", output_dir
    ]

    # 실행
    print(f"[▶] Training Autoformer for {etf} ({pred_len}d)")
    subprocess.run(command)
