# features/generate_features.py

import pandas as pd
import os
import talib
from functools import reduce
from features.price_tech import download_etf_data
from features.sentiment_loader import load_sentiment_data
from preprocessing.lasso_selector import lasso_feature_selection

def make_combined_features0(
    etf_ticker: str,
    start: str,
    end: str,
    sentiment_path: str,
    representatives: list
) -> pd.DataFrame:
    """
    ETF 가격 + 기술지표 + 감정지수를 통합한 단일 DataFrame 생성

    Parameters:
    - etf_ticker (str): ETF 티커 (예: 'XLK')
    - start (str): 시작 날짜 (예: '2018-01-01')
    - end (str): 종료 날짜 (예: '2024-12-31')
    - sentiment_path (str): 감정지수 엑셀 파일 경로
    - representatives (list): ETF에 포함된 종목 티커 (예: ['MSFT US Equity', 'AAPL US Equity'])

    Returns:
    - pd.DataFrame: 날짜 인덱스를 기준으로 병합된 통합 피처
    """
    # 1. ETF 가격 및 기술지표 불러오기
    #price_df = download_etf_data(etf_ticker, start, end)
    #price_df.index = pd.to_datetime(price_df.index)
    #price_df["momentum_3d"] = price_df["close"].pct_change(periods=3)
    print("data generate")
    
    # 2. 감정지수 불러오기
    stock_feature_list = []
    for stock in representatives:
        # 종목별 감정지수 데이터프레임 로드
        stock_senti_df = load_sentiment_data(sentiment_path, stock)
        stock_senti_df.index = pd.to_datetime(stock_senti_df.index)
        stock_senti_df.index = pd.to_datetime(stock_senti_df.index)

        # 시그널 추가
        news_sent = stock_senti_df['news_sentiment']
        twit_sent = stock_senti_df['twitter_sentiment']
        news_heat_z = stock_senti_df['news_heat_z']
        momentum_3d = stock_senti_df["price"].pct_change(periods=3)

        sig1 = ((news_sent > 0.01) & (twit_sent > 0.01) & (news_heat_z > 0.8) & (momentum_3d > 0)).astype(int)
        sig2 = (((stock_senti_df["delta_news"].abs() > 0.02) | (stock_senti_df["delta_twit"].abs() > 0.02)) & (stock_senti_df["delta_ntr"] > 0)).astype(int)
        sig3 = ((stock_senti_df["gap"].abs() > 0.03)).astype(int)
        sig4 = ((news_sent < -0.02) & (momentum_3d < -0.03)).astype(int)
        sig5 = ((stock_senti_df["news_sent_z"] > 0) & (stock_senti_df["news_sent_z"].shift(1) < 0)).astype(int)
        sig6 = ((stock_senti_df["news_sent_z"].shift(2) < -1.0) & (stock_senti_df["news_sent_z"].shift(1) < -0.5) & (stock_senti_df["news_sent_z"] > 0)).astype(int)
        sig7 = (((twit_sent > 0.02) & (news_sent < -0.02)) | ((twit_sent < -0.02) & (news_sent > 0.02))).astype(int)
        sig8 = ((news_heat_z > 1.0) & (stock_senti_df["news_sent_z"] > 0) & (stock_senti_df["news_sent_z"].shift(1) < 0)).astype(int)

        # 시그널을 stock_senti_df에 추가
        stock_senti_df["sig1"] = sig1
        stock_senti_df["sig2"] = sig2
        stock_senti_df["sig3"] = sig3
        stock_senti_df["sig4"] = sig4
        stock_senti_df["sig5"] = sig5
        stock_senti_df["sig6"] = sig6
        stock_senti_df["sig7"] = sig7
        stock_senti_df["sig8"] = sig8

        stock_senti_df['rsi'] = talib.RSI(stock_senti_df['price'].astype(float), timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(stock_senti_df['price'].astype(float), fastperiod=12, slowperiod=26, signalperiod=9)
        stock_senti_df['macd'] = macd
        stock_senti_df['macd_signal'] = macdsignal
        stock_senti_df['macd_hist'] = macdhist
        stock_senti_df['sma_20'] = talib.SMA(stock_senti_df['price'].astype(float), timeperiod=20)
        stock_senti_df["momentum_3d"] = momentum_3d


        # 컬럼명에 종목명 prefix 추가 (예: MSFT_sentiment, ...)
        stock_senti_df = stock_senti_df.add_prefix(f"{stock}_")
        stock_feature_list.append(stock_senti_df)

    # 3. ETF 가격 데이터와 모든 종목별 감정지수 데이터 병합
    all_features = stock_feature_list
    df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="inner"), all_features)

    # 5. 결측치 보간 또는 제거
    df_merged = df_merged.infer_objects(copy=False).interpolate(method="linear").dropna()

    return df_merged

def make_combined_features(etf_ticker: str, sentiment_data: dict, representatives: list) -> pd.DataFrame:
    """
    ETF 가격 + 기술지표 + 감정지수를 통합한 단일 DataFrame 생성

    Parameters:
    - etf_ticker (str): ETF 티커 (예: 'XLK')
    - sentiment_data (dict): ETF별 감정 데이터 딕셔너리
    - representatives (list): ETF에 포함된 종목 티커 (예: ['MSFT US Equity', 'AAPL US Equity'])

    Returns:
    - pd.DataFrame: 날짜 인덱스를 기준으로 병합된 통합 피처
    """
    stock_feature_list = []
    for stock in representatives:
        if stock not in sentiment_data:
            print(f"[⚠️] Sentiment data not found for {stock}. Skipping.")
            continue

        stock_senti_df = sentiment_data[stock]
        stock_senti_df = stock_senti_df.add_prefix(f"{stock}_")
        stock_feature_list.append(stock_senti_df)

    # 모든 종목별 감정지수 데이터 병합
    df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="inner"), stock_feature_list)

    # 결측치 보간 또는 제거
    df_merged = df_merged.infer_objects(copy=False).interpolate(method="linear").dropna()

    return df_merged

def prepare_data(etf_list, sentiment_data, input_dir, processed_dir, target_col, representatives, checking: bool):

    for etf in etf_list:
        processed_file = f"{processed_dir}/{etf}_features.csv"
        autoformer_file = f"{input_dir}/{etf}_autoformer.csv"

        if checking == True : 
            if os.path.exists(processed_file) and os.path.exists(autoformer_file):
                print(f"[INFO] Cached data found for {etf}. Skipping processing.")
                continue

        print(f"[▶] Processing data for {etf}...")
        df_selected, features = lasso_feature_selection(df, target_col=target_col)
        df_selected = make_combined_features(etf_ticker=etf, sentiment_data=df_selected, representatives=representatives[etf])
        df_selected.to_csv(processed_file, index=False)
        calculate_feature_importance(df_selected, target_col=target_col)

        prepare_autoformer_input(input_csv_path=processed_file, output_dir=input_dir, target_col=target_col)
        print(f"[✔] Data processing completed for {etf}.")