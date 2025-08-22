# features/sentiment_loader.py
import pandas as pd

def load_sentiment_data(file_path: str, representative: str) -> pd.DataFrame:
    """
    엑셀 파일에서 감정지수를 로드하고, 대표 종목에 해당하는 감정 데이터만 추출합니다.
    
    Parameters:
    - file_path (str): 감정지수 엑셀 파일 경로
    - representative (str): ETF 대표 종목 티커 (예: 'MSFT US Equity')

    Returns:
    - df (pd.DataFrame): 날짜 index, 뉴스/트위터/뉴스수 포함 감정지수 DataFrame
    """
    xls = pd.ExcelFile(file_path)
    dfs = {sheet: pd.read_excel(file_path, sheet_name=sheet, index_col=0) for sheet in xls.sheet_names}

    # 종목별 데이터 추출
    price = dfs['price'][representative]
    Return = dfs['return'][representative]
    news_sent = dfs['NEWS_SENTIMENT'][representative]
    twit_sent = dfs['TWITTER_SENTIMENT'][representative]
    news_heat = dfs['NEWS_HEAT'][representative]
    neutral_news = dfs['NEWS_NTR_COUNT'][representative]
    news_heat_z = (news_heat - news_heat.rolling(60, min_periods=30).mean()) / news_heat.rolling(60, min_periods=30).std()
    news_sent_z = (news_sent - news_sent.rolling(60, min_periods=30).mean()) / news_sent.rolling(60, min_periods=30).std()

    delta_news = news_sent.diff()
    delta_twit = twit_sent.diff()
    delta_ntr = neutral_news.diff()
    gap = twit_sent - news_sent
    z_diff = news_sent_z - news_sent_z.shift(1)

    # 병합
    df = pd.concat([
        Return.rename("return"),
        price.rename("price"),
        news_sent.rename("news_sentiment"),
        twit_sent.rename("twitter_sentiment"),
        news_heat.rename("news_heat"),
        neutral_news.rename("neutral_news_count"),
        news_heat_z.rename("news_heat_z"),
        news_sent_z.rename("news_sent_z"),
        delta_news.rename("delta_news"),
        delta_twit.rename("delta_twit"),
        delta_ntr.rename("delta_ntr"),
        gap.rename("gap"),
        z_diff.rename("z_diff")
    ], axis=1)

    # 결측치 처리
    df = df.dropna(how="all")
    return df
