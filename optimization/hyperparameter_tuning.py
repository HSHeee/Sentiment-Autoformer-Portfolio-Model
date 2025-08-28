import optuna
from optuna.storages import RDBStorage
from modeling.train_autoformer import train_autoformer
from evaluation.metrics import evaluate_all_etfs

def tune_hyperparameters(etf, input_dir, output_dir, pred_len, target, n_trials=50, n_jobs=4):
    """
    특정 ETF에 대해 Optuna를 사용하여 병렬로 하이퍼파라미터를 튜닝합니다.

    Parameters:
    - etf (str): ETF 이름 (예: "MSFT")
    - input_dir (str): Autoformer 입력 데이터 경로
    - output_dir (str): Autoformer 출력 데이터 경로
    - pred_len (int): 예측 길이
    - target (str): 예측 대상 열 이름
    - n_trials (int): Optuna 탐색 횟수
    - n_jobs (int): 병렬로 실행할 프로세스 수

    Returns:
    - dict: 최적의 하이퍼파라미터
    """
    def objective(trial):
        # 탐색할 하이퍼파라미터 범위 정의
        d_model = trial.suggest_int("d_model", 128, 1024, step=128)
        num_enc_layers = trial.suggest_int("num_enc_layers", 1, 4)
        num_dec_layers = trial.suggest_int("num_dec_layers", 1, 4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

        # Autoformer 학습
        train_autoformer(
            etf=etf,
            input_dir=input_dir,
            output_dir=output_dir,
            pred_len=pred_len,
            target=target,
            hyperparams={
                "d_model": d_model,
                "num_enc_layers": num_enc_layers,
                "num_dec_layers": num_dec_layers,
                "learning_rate": learning_rate,
            }
        )

        # 평가 지표 계산 (예: MSE)
        results = evaluate_all_etfs([etf], model_name="Autoformer", output_root=output_dir)
        return results[etf]["MSE"]  # MSE 값을 반환
    
    print(f"[▶] Tuning hyperparameters for {etf}")
    # RDBStorage를 사용하여 병렬 처리 설정
    storage = RDBStorage(url="sqlite:///optuna_study.db")  # SQLite 데이터베이스 사용
    study = optuna.create_study(direction="minimize", storage=storage, study_name=f"{etf}_study", load_if_exists=True)

    # 병렬로 탐색 실행
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    print(f"[✔] Best hyperparameters for {etf}: {study.best_params}")

    return study.best_params

