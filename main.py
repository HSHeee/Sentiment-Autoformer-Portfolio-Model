from features.price_tech import download_etf_data

if __name__ == "__main__":
    xlk_df = download_etf_data("XLK", "2018-01-01", "2024-12-31")
    print(xlk_df.head())