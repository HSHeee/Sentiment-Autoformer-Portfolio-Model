import torch
from features.price_tech import download_etf_data
from features.sentiment_loader import load_sentiment_data
from preprocessing.merge_features import merge_price_and_sentiment
from modeling.lasso_selector import generate_targets, lasso_feature_selection
from modeling.train_loop import train_model_autoformer

from features.generate_features import make_combined_features
from preprocessing.lasso_selector import lasso_feature_selection
from preprocessing.input_transform import prepare_autoformer_input
from modeling.train_autoformer import train_autoformer
from evaluation.metrics import evaluate_all_etfs


"""
if __name__ == "__main__":
    xlk_price = download_etf_data("XLK", "2018-01-01", "2024-12-31")
    msft_sentiment = load_sentiment_data("data/senti&price.xlsx", "MSFT US Equity")  # XLK용
    #xom_sentiment = load_sentiment_data(file_path, "XOM US Equity")   # XLE용
    #jpm_sentiment = load_sentiment_data(file_path, "JPM US Equity")   # XLF용

    xlk_merged = merge_price_and_sentiment(xlk_price, msft_sentiment)
    print(xlk_merged.head())

    target = generate_targets(xlk_merged, target_col="close", horizon=1)
    X = xlk_merged.drop(columns=["close", "open", "high", "low", "volume", "ticker"])  # raw price 제외
    X_selected  = lasso_feature_selection(X, target)
    print("선택된 feature 수:", X_selected .shape[1])
    print("선택된 feature 목록:", X_selected .columns.tolist())
    X_selected = X_selected.dropna()
    target = target.loc[X_selected.index]
    
    model = train_model_autoformer(
        X=X_selected.values,
        y=target.values,
        input_dim=X_selected.shape[1],
        enc_len=30,
        label_len=15,
        pred_len=5,
        epochs=20,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )"""

    
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
