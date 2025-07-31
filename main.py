from features.generate_features import make_combined_features
from preprocessing.lasso_selector import lasso_feature_selection
from preprocessing.input_transform import prepare_autoformer_input
from modeling.train_autoformer import train_autoformer
from evaluation.metrics import evaluate_all_etfs
from evaluation.backtest import *



if __name__ == "__main__":
    REPRESENTATIVES = {
    "XLK": "MSFT US Equity",
    "XLE": "XOM US Equity",
    "XLF": "JPM US Equity"
    }
    etf_list = ["XLK"]
    
    for etf in etf_list:
        df = make_combined_features(
            etf_ticker=etf,
            start="2018-01-01",
            end="2024-12-31",
            sentiment_path="data/senti&price.xlsx",
            representative=REPRESENTATIVES[etf]
        )
        df = df.drop(columns=["ticker"])
        df_selected, features = lasso_feature_selection(df, target_col="close")
        df_selected.to_csv(f"data/processed/{etf}_features.csv")

    for etf in etf_list:
        prepare_autoformer_input(
            input_csv_path=f"data/processed/{etf}_features.csv",
            output_dir="data/autoformer_input",
            target_col="close"
        )

    for etf in etf_list:
        train_autoformer(
            etf=etf,
            input_dir="data/autoformer_input",
            output_dir=f"outputs/{etf}",
            pred_len=5,
            target="close"
        )

    results = evaluate_all_etfs(etf_list, output_root="outputs")

    summary = backtest_all(etf_list)
    print(summary)

    models = ["Autoformer", "LSTM", "GRU", "Linear"]
    summary_df = backtest_Model_all(etf_list, models)
    print(summary_df)