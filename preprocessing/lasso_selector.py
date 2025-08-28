# preprocessing/lasso_selector.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

def lasso_feature_selection(df: pd.DataFrame, target_col: str, alpha_range=(0.0001, 1.0)) -> tuple:
    """
    LASSO로 피처 선택 수행

    Parameters:
    - df (pd.DataFrame): 통합 피처 + 타겟 포함된 DataFrame
    - target_col (str): 예측할 타겟 변수 이름 (ex: 'Close', 'return', etc.)
    - alpha_range (tuple): LASSO alpha 탐색 범위

    Returns:
    - selected_df (pd.DataFrame): 선택된 피처만 포함한 정규화 DataFrame
    - selected_features (List[str]): 선택된 피처 이름 목록
    """
    # 1. X, y 분리
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. LASSO with cross-validation
    model = LassoCV(alphas=None, cv=5, random_state=42, max_iter=5000).fit(X_scaled, y)

    # 4. 계수 기준으로 피처 선택
    coef = pd.Series(model.coef_, index=X.columns)
    selected_features = coef[coef != 0].index.tolist()

    # 선택된 feature와 중요도 출력
    print("Selected Features:", selected_features)
    print("Feature Importance:", coef[coef != 0])

    # 5. 선택된 피처만 정규화 데이터로 반환
    selected_df = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)[selected_features]
    selected_df[target_col] = y.values  # 타겟 복원

    return selected_df, selected_features

def calculate_feature_importance(df: pd.DataFrame, target_col: str):
    """
    LightGBM/XGBoost를 사용하여 feature importance 계산 및 시각화
    """
    # 데이터 분리
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 모델 학습
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    # Feature Importance 계산
    importance = model.feature_importances_
    feature_names = X.columns

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance (GradientBoosting)')
    plt.show()

    return importance
