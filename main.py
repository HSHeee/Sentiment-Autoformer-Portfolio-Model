from features.generate_features import make_combined_features
from preprocessing.lasso_selector import lasso_feature_selection, calculate_feature_importance
from preprocessing.input_transform import prepare_autoformer_input
from modeling.train_autoformer import train_autoformer
from evaluation.metrics import evaluate_all_etfs
from evaluation.backtest import *
from optimization.hyperparameter_tuning import tune_hyperparameters
from utils.parallel_training import *


if __name__ == "__main__":
    REPRESENTATIVES2 = {
    "XLK": ["MSFT US Equity", "AAPL US Equity", "GOOGL US Equity"],
    "XLF": ["JPM US Equity", "V US Equity"],
    }
    etf_list2 = ["XLK","XLF"]

    REPRESENTATIVES = {
    "MSFT": ["MSFT US Equity"],
    "AAPL": ["AAPL US Equity"],
    "GOOGL": ["GOOGL US Equity"],
    "ORCL": ["ORCL US Equity"],
    "AMZN": ["AMZN US Equity"],
    "META": ["META US Equity"],
    "NFLX": ["NFLX US Equity"],
    "JPM": ["JPM US Equity"],
    "V": ["V US Equity"],
    }
    etf_list = ["ORCL","AMZN"] #"MSFT","AAPL","GOOGL",,"META","NFLX","JPM","V"
    pred_len = 1
    TARGET = "return"
    input_dir = "data/autoformer_input"
    output_dir = "outputs"
    n_trials = 10
    n_jobs = 1 # 병렬로 돌릴 개수
    gpu_ids = [0]  # 사용 가능한 GPU ID 리스트
    
    for etf in etf_list:
        constituents = REPRESENTATIVES[etf]
        df = make_combined_features(
            etf_ticker=etf,
            start="2018-01-01",
            end="2025-07-30",
            sentiment_path="data/senti&price.xlsx",
            representatives=constituents
        )

        df = df.drop(columns=["ticker", "close", "open", "high", "low", "volume","return_5d"])
        df = df.dropna()
 
        df_selected, features = lasso_feature_selection(df, target_col=TARGET)
        df_selected.to_csv(f"data/processed/{etf}_features.csv")
        calculate_feature_importance(df_selected, target_col=TARGET)

    for etf in etf_list:
        prepare_autoformer_input(
            input_csv_path=f"data/processed/{etf}_features.csv",
            output_dir=input_dir,
            target_col=TARGET
        )

    # 2. 각 ETF별 하이퍼파라미터 튜닝
    best_params_per_etf = parallel_tune(etf_list, input_dir, output_dir, pred_len, TARGET, n_trials, n_jobs, gpu_ids)

    print("BBBBBBBBBBBBBBBBBeest parma :", best_params_per_etf)
    parallel_train(etf_list, input_dir, output_dir, pred_len, TARGET, best_params_per_etf, gpu_ids)


    # 3. 병렬로 학습 실행
    parallel_train(etf_list, input_dir, output_dir, pred_len, TARGET, best_params_per_etf, gpu_ids)

    results = evaluate_all_etfs(etf_list, model_name="Autoformer", output_root=output_dir)
    print("[✔] Final evaluation results:", results)

    summary = backtest_all(
    etf_list,
    short_enabled=False, #True
    full_position=True,
    threshold=0.01,
    hold_days=2,        # or 3
    k_consecutive=None,    # or 2
    cost_bps=0,
    plot_trades=True       # 리눅스에서 화면 없으면 png로 저장
    )
    print(summary)
    
    #models = ["Autoformer", "LSTM", "GRU", "Linear"]
    #summary_df = backtest_Model_all(etf_list, models)
    #print(summary_df)