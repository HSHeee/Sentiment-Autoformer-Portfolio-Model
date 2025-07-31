# preprocessing/merge_features.py
import pandas as pd

def merge_price_and_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    ETF의 가격 및 기술지표 데이터와 해당 ETF를 대표하는 종목의 감정지수를 날짜 기준으로 병합합니다.

    Parameters:
    - price_df (pd.DataFrame): ETF 가격 및 기술지표 데이터 (index: Date)
    - sentiment_df (pd.DataFrame): 대표 종목의 감정지수 (index: Date)

    Returns:
    - merged_df (pd.DataFrame): 병합된 DataFrame
    """
    # 인덱스 날짜 형태
    price_df.index = pd.to_datetime(price_df.index)
    sentiment_df.index = pd.to_datetime(sentiment_df.index)

    # 멀티인덱스일 경우 첫 번째 레벨만 사용
    if isinstance(price_df.index, pd.MultiIndex):
        price_df.index = price_df.index.get_level_values(0)
    # 컬럼이 MultiIndex면 평탄화
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)


    # 날짜 기준 left join
    merged_df = price_df.merge(sentiment_df, how='left', left_index=True, right_index=True)

    # 결측치는 forward fill, 필요시 dropna도 가능
    merged_df = merged_df.ffill()

    return merged_df
