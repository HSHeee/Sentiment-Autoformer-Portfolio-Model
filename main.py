from features.price_tech import download_etf_data
from features.sentiment_loader import load_sentiment_data
from preprocessing.merge_features import merge_price_and_sentiment

if __name__ == "__main__":
    xlk_df = download_etf_data("XLK", "2018-01-01", "2024-12-31")
    
    file_path = "data/senti&price.xlsx"
    msft_sentiment = load_sentiment_data(file_path, "MSFT US Equity")  # XLK용
    
    #xom_sentiment = load_sentiment_data(file_path, "XOM US Equity")   # XLE용
    #jpm_sentiment = load_sentiment_data(file_path, "JPM US Equity")   # XLF용

    xlk_merged = merge_price_and_sentiment(xlk_df, msft_sentiment)

    print(xlk_merged.head())
    