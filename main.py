from features.sentiment_loader import load_sentiment_data
from features.generate_features import make_combined_features, prepare_data
from preprocessing.lasso_selector import lasso_feature_selection, calculate_feature_importance
from preprocessing.input_transform import prepare_autoformer_input
from modeling.train_autoformer import train_autoformer
from evaluation.metrics import evaluate_all_etfs
from evaluation.backtest import *
from optimization.hyperparameter_tuning import tune_hyperparameters
from utils.parallel_training import *


if __name__ == "__main__":
    REPRESENTATIVES = {
        "MSFT": "MSFT US Equity",
        "AAPL": "AAPL US Equity",
        "GOOGL": "GOOGL US Equity",
        "ORCL": "ORCL US Equity",
        "AMZN": "AMZN US Equity",
        "META": "META US Equity",
        "NFLX": "NFLX US Equity",
        "JPM": "JPM US Equity",
        "V": "V US Equity",
    }
    etf_list = ["MSFT"] #,"AAPL","GOOGL","ORCL","AMZN","META","NFLX","JPM","V"
    pred_len = 1
    TARGET = "return"
    sentiment_path ="data/senti&price.xlsx"
    input_dir = "data/autoformer_input"
    processed_dir = "data/processed"
    output_dir = "outputs"
    n_trials = 10
    n_jobs = 1 # 병렬로 돌릴 개수
    gpu_ids = [0]  # 사용 가능한 GPU ID 리스트

    # 데이터 준비
    #sentiment_data = load_sentiment_data(sentiment_path, processed_dir, etf_list, REPRESENTATIVES)
    sentiment_data = {}
    for etf in etf_list:
        df = pd.read_csv(processed_dir + f"/{etf}_features.csv")
        sentiment_data[etf] = df
    print("processed finish")

    #prepare_autoformer_input(etf_list, sentiment_data, input_dir, TARGET)

    # 2. 각 ETF별 하이퍼파라미터 튜닝
    #best_params_per_etf = parallel_tune(etf_list, input_dir, output_dir, pred_len, TARGET, n_trials, n_jobs, gpu_ids)
    best_params_per_etf = {
        'MSFT': {'d_model': 384, 'num_enc_layers': 3, 'num_dec_layers': 2, 'learning_rate': 7.447809323751878e-05},
        'AAPL': {'d_model': 640, 'num_enc_layers': 4, 'num_dec_layers': 1, 'learning_rate': 0.0015922121189894647},
        'GOOGL': {'d_model': 256, 'num_enc_layers': 1, 'num_dec_layers': 3, 'learning_rate': 0.00015079626759664257},
        'ORCL': {'d_model': 384, 'num_enc_layers': 2, 'num_dec_layers': 2, 'learning_rate': 0.0003962772798271517},
        'AMZN': {'d_model': 512, 'num_enc_layers': 1, 'num_dec_layers': 1, 'learning_rate': 0.0012308600768460738},
        'META': {'d_model': 896, 'num_enc_layers': 3, 'num_dec_layers': 1, 'learning_rate': 1.9001126251449003e-05},
        'NFLX': {'d_model': 384, 'num_enc_layers': 4, 'num_dec_layers': 2, 'learning_rate': 0.0004357912428052041},
        'JPM': {'d_model': 384, 'num_enc_layers': 4, 'num_dec_layers': 1, 'learning_rate': 0.001319618751414433},
        'V': {'d_model': 640, 'num_enc_layers': 3, 'num_dec_layers': 1, 'learning_rate': 0.0006643355967680847}
    }
    print("BBBBBBBBBBBBBBBBBeest parma :", best_params_per_etf)

    # 3. 병렬로 학습 실행
    parallel_train(etf_list, input_dir, output_dir, pred_len, TARGET, best_params_per_etf, gpu_ids)

    results = evaluate_all_etfs(etf_list, model_name="Autoformer", output_root=output_dir)
    print("[✔] Final evaluation results:", results)

    summary = backtest_all(
    etf_list,
    short_enabled=True, #True
    full_position=True,
    threshold=0.001,
    hold_days=2,        # or 3
    k_consecutive=None,    # or 2
    cost_bps=0,
    plot_trades=True       # 리눅스에서 화면 없으면 png로 저장
    )
    print(summary)
    
    #models = ["Autoformer", "LSTM", "GRU", "Linear"]
    #summary_df = backtest_Model_all(etf_list, models)
    #print(summary_df)