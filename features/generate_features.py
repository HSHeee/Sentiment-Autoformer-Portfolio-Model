# features/generate_features.py

import pandas as pd
from features.price_tech import download_etf_data
from features.sentiment_loader import load_sentiment_data

def make_combined_features(
    etf_ticker: str,
    start: str,
    end: str,
    sentiment_path: str,
    representative: str
) -> pd.DataFrame:
    """
    ETF 가격 + 기술지표 + 감정지수를 통합한 단일 DataFrame 생성

    Parameters:
    - etf_ticker (str): ETF 티커 (예: 'XLK')
    - start (str): 시작 날짜 (예: '2018-01-01')
    - end (str): 종료 날짜 (예: '2024-12-31')
    - sentiment_path (str): 감정지수 엑셀 파일 경로
    - representative (str): ETF를 대표하는 개별 종목 티커 (예: 'MSFT US Equity')

    Returns:
    - pd.DataFrame: 날짜 인덱스를 기준으로 병합된 통합 피처
    """
    # 1. ETF 가격 및 기술지표 불러오기
    price_df = download_etf_data(etf_ticker, start, end)

    # 2. 감정지수 불러오기
    senti_df = load_sentiment_data(sentiment_path, representative)

    # 3. 날짜 인덱스 정리 및 병합
    price_df.index = pd.to_datetime(price_df.index)
    senti_df.index = pd.to_datetime(senti_df.index)

    df_merged = pd.merge(price_df, senti_df, left_index=True, right_index=True, how="inner")

    # 4. 결측치 보간 또는 제거
    df_merged = df_merged.interpolate(method="linear").dropna()

    return df_merged
