import pandas as pd
import os

def prepare_autoformer_input(
    input_csv_path: str,
    output_dir: str,
    target_col: str = "close"
):
    """
    Autoformer 입력 포맷으로 변환하여 저장

    Parameters:
    - input_csv_path (str): LASSO 선택 후 저장된 CSV 파일 경로
    - output_dir (str): 저장할 디렉토리
    - target_col (str): 예측 대상 컬럼명 (예: 'close')
    """
    df = pd.read_csv(input_csv_path, index_col=0, parse_dates=True)

    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)

    # 날짜순 정렬
    df = df.sort_values("date")

    # 저장 경로 및 이름 구성
    etf_name = os.path.basename(input_csv_path).split("_")[0]
    output_path = os.path.join(output_dir, f"{etf_name}_autoformer.csv")

    df.to_csv(output_path, index=False)
    print(f"[✔] Saved Autoformer input: {output_path}")
