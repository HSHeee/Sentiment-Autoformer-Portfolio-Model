# features/price_tech.py
# ETF 가격 및 기술지표를 Yahoo Finance에서 가져오는 모듈
import yfinance as yf
import pandas_ta as ta
import pandas as pd

def download_etf_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    주어진 ticker의 ETF 데이터를 Yahoo Finance에서 다운로드하고
    pandas-ta 기반의 기술지표를 포함한 DataFrame을 반환합니다.

    Parameters:
    - ticker (str): 예) 'XLK'
    - start (str): 시작일 예) '2018-01-01'
    - end (str): 종료일 예) '2024-12-31'

    Returns:
    - df (pd.DataFrame): OHLCV + 기술지표 포함 DataFrame
    """
    df = yf.download(ticker, start=start, end=end)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })
    df = df.ffill()
    #df.ta.strategy(name="All", core=True)  # MA, RSI, MACD 등 주요 기술지표 자동 생성
    df.ta.rsi(inplace=True)
    df.ta.macd(inplace=True)
    df.ta.sma(length=20, inplace=True)
    df["ticker"] = ticker
    return df