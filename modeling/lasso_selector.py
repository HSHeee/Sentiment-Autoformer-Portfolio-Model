# modeling/lasso_selector.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def generate_targets(df: pd.DataFrame, target_col: str = "close", horizon: int = 1) -> pd.Series:
    """
    예측 목표 변수 생성. 종가, 수익률, 방향성 중 선택 가능.

    Parameters:
    - df (pd.DataFrame): 입력 데이터
    - target_col (str): 예측 대상 컬럼 (예: 'close')
    - horizon (int): 예측 시차 (예: T+1 → 1)

    Returns:
    - target (pd.Series): 목표 변수 (float 또는 0/1)
    """
    future = df[target_col].shift(-horizon)
    returns = (future - df[target_col]) / df[target_col]

    if target_col == "direction":
        return (returns > 0).astype(int)
    else:
        return returns if target_col == "return" else future


def lasso_feature_selection(df: pd.DataFrame, target: pd.Series, alpha_grid=None, random_state=42) -> pd.DataFrame:
    """
    LASSO 회귀로 중요한 feature 선택

    Parameters:
    - df (pd.DataFrame): feature 데이터프레임
    - target (pd.Series): 예측할 타겟 시리즈
    - alpha_grid (list or np.ndarray): 알파 후보군 (자동 탐색)
    - random_state (int): 시드

    Returns:
    - selected_df (pd.DataFrame): LASSO로 선택된 feature만 포함된 df
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 0, 50)

    # 결측 제거
    merged = df.copy()
    merged["target"] = target
    merged = merged.dropna()

    X = merged.drop("target", axis=1)
    y = merged["target"]

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LASSO
    lasso = LassoCV(alphas=alpha_grid, cv=5, random_state=random_state).fit(X_scaled, y)
    coef = pd.Series(lasso.coef_, index=X.columns)

    selected_features = coef[coef != 0].index.tolist()
    selected_df = df[selected_features]

    return selected_df
