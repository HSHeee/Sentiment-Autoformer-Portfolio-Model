from features.generate_features import make_combined_features
from preprocessing.lasso_selector import lasso_feature_selection, calculate_feature_importance
from preprocessing.input_transform import prepare_autoformer_input
from modeling.train_autoformer import train_autoformer
from evaluation.metrics import evaluate_all_etfs
from evaluation.backtest import *



if __name__ == "__main__":
    REPRESENTATIVES = {
    "XLK": ["MSFT US Equity", "AAPL US Equity", "GOOGL US Equity"],
    "XLE": ["XOM US Equity"],
    "XLF": ["JPM US Equity", "V US Equity"],
    }
    etf_list = ["XLK","XLF"]
    pred_len = 1
    TARGET = "return"
    
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
            output_dir="data/autoformer_input",
            target_col=TARGET
        )

    for etf in etf_list:
        train_autoformer(
            etf=etf,
            input_dir="data/autoformer_input",
            output_dir=f"outputs",
            pred_len=pred_len,
            target=TARGET
        )

    results = evaluate_all_etfs(etf_list, model_name="Autoformer", output_root="outputs")

    summary = backtest_all(etf_list, short_enabled=True)
    print(summary)
    
    models = ["Autoformer", "LSTM", "GRU", "Linear"]
    #summary_df = backtest_Model_all(etf_list, models)
    #print(summary_df)