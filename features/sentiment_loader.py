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
    news_sent = dfs['NEWS_SENTIMENT'][representative]
    twit_sent = dfs['TWITTER_SENTIMENT'][representative]
    news_heat = dfs['NEWS_HEAT'][representative]
    neutral_news = dfs['NEWS_NTR_COUNT'][representative]

    # 병합
    df = pd.concat([
        news_sent.rename("news_sentiment"),
        twit_sent.rename("twitter_sentiment"),
        news_heat.rename("news_heat"),
        neutral_news.rename("neutral_news_count")
    ], axis=1)

    # 결측치 처리
    df = df.dropna(how="all")
    return df
