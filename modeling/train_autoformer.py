import sys
sys.path.append("../Autoformer")
import os
import subprocess
import numpy as np
import pandas as pd

def train_autoformer(
    etf: str,
    input_dir: str = "data/autoformer_input",
    output_dir: str = "outputs",
    model_name = "Autoformer",
    pred_len: int = 1,
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
        "python", "../Autoformer/run.py",
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
        "--train_epochs", "5",     # 추후 조정 필요
        "--batch_size", "32",
        "--learning_rate", "0.001",
        "--patience", "2",
        "--checkpoints", output_dir
    ]

    # 실행
    print(f"[▶] Training Autoformer for {etf} ({pred_len}d)")
    subprocess.run(command)

    # setting 문자열 생성 규칙과 동일하게 맞춰야 함
    setting = f"{etf}_{pred_len}d_{model_name}_custom_ftM_sl60_ll30_pl{pred_len}_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"
    results_dir = os.path.join("results", setting)
    pred_path = os.path.join(results_dir, "pred.npy")
    true_path = os.path.join(results_dir, "true.npy")
    csv_path = os.path.join(output_dir, model_name, f"{etf}_prediction.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    """if os.path.exists(pred_path) and os.path.exists(true_path):
        preds = np.load(pred_path)
        trues = np.load(true_path)
        # (N, 5, 1) -> (N*5, )
        preds = preds.reshape(-1)
        trues = trues.reshape(-1)
        df_input = pd.read_csv(os.path.join(input_dir, data_path))
        date_col = df_input["date"].values[-len(preds):]
        df = pd.DataFrame({"date": date_col, "true": trues, "pred": preds})
        df.to_csv(csv_path, index=False)
        print(f"[✔] prediction.csv saved: {csv_path}")
    else:
        print(f"[⚠️] pred.npy or true.npy not found in {results_dir}")"""
    
    #임시 해결용 코드 ... 
    if os.path.exists(pred_path) and os.path.exists(true_path):
        preds = np.load(pred_path)  # (N, pred_len, 1)
        trues = np.load(true_path)  # (N, pred_len, 1)
        N, pred_len, _ = preds.shape
        preds = preds.reshape(-1)
        trues = trues.reshape(-1)

        df_input = pd.read_csv(os.path.join(input_dir, data_path))
        date_arr = pd.to_datetime(df_input["date"].values)

        seq_len = 60  # 반드시 run.py와 동일하게!
        label_len = 30  # 반드시 run.py와 동일하게!
        # test 샘플 시작 인덱스: train+val 이후부터 test set sliding 시작
        # test set의 시작 인덱스는 전체 데이터 길이 - (N + seq_len + label_len + pred_len - 1)
        # 하지만 일반적으로 test set은 마지막 N개 window에서 시작

        # test 샘플의 시작 인덱스들
        test_start_idxs = np.arange(len(date_arr) - N - pred_len + 1, len(date_arr) - pred_len + 1)
        # 각 샘플별 예측 날짜
        date_col = []
        for idx in test_start_idxs:
            # 각 샘플의 예측 구간 날짜: seq_len + label_len 이후부터 pred_len개
            pred_dates = date_arr[idx + seq_len + label_len : idx + seq_len + label_len + pred_len]
            date_col.extend(pred_dates)
        date_col = np.array(date_col)

        # 길이 맞추기 (혹시라도 오버플로우 방지)
        min_len = min(len(date_col), len(preds), len(trues))
        date_col = date_col[:min_len]
        preds = preds[:min_len]
        trues = trues[:min_len]

        df = pd.DataFrame({"date": date_col, "true": trues, "pred": preds})
        df.to_csv(csv_path, index=False)
        print(f"[✔] prediction.csv saved: {csv_path}")
    else:
        print(f"[⚠️] pred.npy or true.npy not found in {results_dir}")