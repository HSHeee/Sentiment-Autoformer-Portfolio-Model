# features/sentiment_loader.py
import pandas as pd
import os
import talib

def calculate_signals(news_sent, twit_sent, news_heat_z, momentum_3d, delta_news, delta_twit, delta_ntr, gap, news_sent_z):
        """
        신호 계산 로직을 함수로 분리
        """
        signals = {
            "sig1": ((news_sent > 0.01) & (twit_sent > 0.01) & (news_heat_z > 0.8) & (momentum_3d > 0)).astype(int),
            "sig2": (((delta_news.abs() > 0.02) | (delta_twit.abs() > 0.02)) & (delta_ntr > 0)).astype(int),
            "sig3": ((gap.abs() > 0.03)).astype(int),
            "sig4": ((news_sent < -0.02) & (momentum_3d < -0.03)).astype(int),
            "sig5": ((news_sent_z > 0) & (news_sent_z.shift(1) < 0)).astype(int),
            "sig6": ((news_sent_z.shift(2) < -1.0) & (news_sent_z.shift(1) < -0.5) & (news_sent_z > 0)).astype(int),
            "sig7": (((twit_sent > 0.02) & (news_sent < -0.02)) | ((twit_sent < -0.02) & (news_sent > 0.02))).astype(int),
            "sig8": ((news_heat_z > 1.0) & (news_sent_z > 0) & (news_sent_z.shift(1) < 0)).astype(int),
        }
        return pd.DataFrame(signals)

def preprocess_data(dfs, etf_values):
        """
        개별 ETF 데이터를 전처리하고 날짜를 기준으로 길이를 맞춥니다.
        """
        # 필요한 데이터 추출
        price = dfs['price'][etf_values]
        Return = dfs['Return'][etf_values]
        news_sent = dfs['NEWS_SENTIMENT'][etf_values]
        twit_sent = dfs['TWITTER_SENTIMENT'][etf_values]
        news_heat = dfs['NEWS_HEAT'][etf_values]
        neutral_news = dfs['NEWS_NTR_COUNT'][etf_values]

        # 공통 날짜 인덱스 생성
        common_dates = price.index.union(news_sent.index).union(twit_sent.index)

        # 모든 데이터프레임을 공통 날짜 인덱스에 맞춤
        price = price.reindex(common_dates)
        Return = Return.reindex(common_dates)
        news_sent = news_sent.reindex(common_dates)
        twit_sent = twit_sent.reindex(common_dates)
        news_heat = news_heat.reindex(common_dates)
        neutral_news = neutral_news.reindex(common_dates)

        # 결측치 처리 (선형 보간법 사용)
        price = price.interpolate(method="linear")
        Return = Return.interpolate(method="linear")
        news_sent = news_sent.interpolate(method="linear")
        twit_sent = twit_sent.interpolate(method="linear")
        news_heat = news_heat.interpolate(method="linear")
        neutral_news = neutral_news.interpolate(method="linear")

        return price, Return, news_sent, twit_sent, news_heat, neutral_news

def load_sentiment_data(file_path: str, processed_dir: str, representatives: list, REPRESENTATIVES:dict) -> dict:
    """
    엑셀 파일에서 모든 ETF의 감정 데이터를 로드합니다.

    Parameters:
    - file_path (str): 감정지수 엑셀 파일 경로

    Returns:
    - dict: ETF별 감정 데이터 DataFrame 딕셔너리
    """
    xls = pd.ExcelFile(file_path)
    dfs = {sheet: pd.read_excel(file_path, sheet_name=sheet, index_col=0) for sheet in xls.sheet_names}
    print("df load.")
    etf_data = {}

    for representative in representatives:
        etf_values = REPRESENTATIVES.get(representative)
        price, Return, news_sent, twit_sent, news_heat, neutral_news = preprocess_data(dfs, etf_values)

        news_heat_z = (news_heat - news_heat.rolling(60, min_periods=30).mean()) / news_heat.rolling(60, min_periods=30).std()
        news_sent_z = (news_sent - news_sent.rolling(60, min_periods=30).mean()) / news_sent.rolling(60, min_periods=30).std()
        momentum_3d = price.pct_change(periods=3)

        delta_news = news_sent.diff()
        delta_twit = twit_sent.diff()
        delta_ntr = neutral_news.diff()
        gap = twit_sent - news_sent
        z_diff = news_sent_z - news_sent_z.shift(1)

        signals = calculate_signals(news_sent, twit_sent, news_heat_z, momentum_3d, delta_news, delta_twit, delta_ntr, gap, news_sent_z)

        rsi = pd.Series(talib.RSI(price.astype(float).values, timeperiod=14), index=price.index, name="rsi")
        macd_arr, macd_signal_arr, macd_hist_arr = talib.MACD(price.astype(float).values, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = pd.Series(macd_arr, index=price.index, name="macd")
        macd_signal = pd.Series(macd_signal_arr, index=price.index, name="macd_signal")
        macd_hist = pd.Series(macd_hist_arr, index=price.index, name="macd_hist")
        sma_20 = pd.Series(talib.SMA(price.astype(float).values, timeperiod=20), index=price.index, name="sma_20")

        # 병합: concat 대신 merge를 사용하여 날짜를 기준으로 병합
        df = signals.merge(price.to_frame("price"), left_index=True, right_index=True, how="outer")
        df = df.merge(Return.to_frame("return"), left_index=True, right_index=True, how="outer")
        df = df.merge(news_sent.to_frame("news_sent"), left_index=True, right_index=True, how="outer")
        df = df.merge(twit_sent.to_frame("twit_sent"), left_index=True, right_index=True, how="outer")
        df = df.merge(news_heat.to_frame("news_heat"), left_index=True, right_index=True, how="outer")
        df = df.merge(neutral_news.to_frame("neutral_news"), left_index=True, right_index=True, how="outer")
        df = df.merge(news_heat_z.to_frame("news_heat_z"), left_index=True, right_index=True, how="outer")
        df = df.merge(news_sent_z.to_frame("news_sent_z"), left_index=True, right_index=True, how="outer")
        df = df.merge(delta_news.to_frame("delta_news"), left_index=True, right_index=True, how="outer")
        df = df.merge(delta_twit.to_frame("delta_twit"), left_index=True, right_index=True, how="outer")
        df = df.merge(delta_ntr.to_frame("delta_ntr"), left_index=True, right_index=True, how="outer")
        df = df.merge(gap.to_frame("gap"), left_index=True, right_index=True, how="outer")
        df = df.merge(z_diff.to_frame("z_diff"), left_index=True, right_index=True, how="outer")

        # TALIB 부분도 Series에 name을 지정했으니 그대로 to_frame()만:
        df = df.merge(rsi.to_frame(), left_index=True, right_index=True, how="outer")
        df = df.merge(macd.to_frame(), left_index=True, right_index=True, how="outer")
        df = df.merge(macd_signal.to_frame(), left_index=True, right_index=True, how="outer")
        df = df.merge(macd_hist.to_frame(), left_index=True, right_index=True, how="outer")
        df = df.merge(sma_20.to_frame(), left_index=True, right_index=True, how="outer")

        # 결측치 처리
        df = df.interpolate(method="linear")  # 선형 보간법으로 NaN 값 채우기

        # 결측치 처리
        df = df.dropna(how="any")
        etf_data[representative] = df
        output_path = os.path.join(processed_dir, f"{representative}_features.csv")
        df.to_csv(output_path, index=True)
        print(f"[✔] Saved feature: {processed_dir}/{representative}_features.csv")
    return etf_data
