from features.price_tech import download_etf_data
from features.sentiment_loader import load_sentiment_data
from preprocessing.merge_features import merge_price_and_sentiment
from modeling.lasso_selector import generate_targets, lasso_feature_selection
from modeling.train_loop import train_model


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


    model = model = train_model(X_selected.values, target.values, input_dim=X_selected.shape[1])

    