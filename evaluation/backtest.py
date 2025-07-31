import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def simple_backtest(pred_path: str, price_path: str, etf: str = "") -> pd.DataFrame:
    """
    예측 결과를 바탕으로 단순 전략 백테스트 수행

    Parameters:
    - pred_path (str): prediction.csv 경로 (Autoformer 출력)
    - price_path (str): 실제 ETF 가격 CSV 경로 (종가 포함)
    - etf (str): ETF 이름 (Optional, 로그 출력용)

    Returns:
    - pd.DataFrame: 전략 vs 벤치마크 누적 수익률 비교 테이블
    """
    # 1. Autoformer 예측 불러오기
    pred_df = pd.read_csv(pred_path)
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    price_df = price_df[~price_df.index.duplicated(keep="first")]
    price_df = price_df.sort_index()

    # 날짜 기준 merge
    if "date" in pred_df.columns:
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        price_df = price_df.reset_index().rename(columns={"index": "date"})
        merged = pd.merge(pred_df, price_df, on="date", how="inner")
        y_true = merged["true"].values
        y_pred = merged["pred"].values
        ret = merged["close"].pct_change().fillna(0).values
    else:
        # 기존 방식 fallback
        y_true = pred_df["true"].values
        y_pred = pred_df["pred"].values
        start_idx = -len(y_true)
        ret = price_df["close"].pct_change().fillna(0).values[start_idx:]

    signal = (y_pred > y_true).astype(int)
    strategy_ret = signal * ret

    strategy_cum = np.cumprod(1 + strategy_ret)
    bench_cum = np.cumprod(1 + ret)

    result_df = pd.DataFrame({
        "Strategy": strategy_cum,
        "Benchmark": bench_cum
    }, index=merged["date"] if "date" in pred_df.columns else price_df.index[start_idx:])

    # 7. 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(result_df.index, result_df["Strategy"], label="Strategy")
    plt.plot(result_df.index, result_df["Benchmark"], label="Benchmark", linestyle="--")
    plt.title(f"{etf} Autoformer 전략 수익률")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return result_df

def compute_risk_metrics(ret: np.ndarray) -> dict:
    """
    리스크 지표 계산 (Sharpe ratio, MDD, Volatility)

    Parameters:
    - ret: 일간 수익률 (np.ndarray)

    Returns:
    - dict: 지표 결과
    """
    cum = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    mdd = drawdown.min()

    sharpe = np.mean(ret) / (np.std(ret) + 1e-8) * np.sqrt(252)
    vol = np.std(ret) * np.sqrt(252)

    return {
        "Sharpe": sharpe,
        "Volatility": vol,
        "Max Drawdown": mdd
    }


def backtest_all(etf_list, output_root="outputs", price_root="data/processed"):
    """
    여러 ETF를 반복 백테스트하고 수익률/리스크 지표 비교

    Parameters:
    - etf_list: 예측한 ETF 티커 리스트
    - output_root: prediction.csv 위치 기준
    - price_root: 실제 ETF 가격 (정규화된 features.csv 경로)

    Returns:
    - pd.DataFrame: ETF별 성과 요약 테이블
    """
    result_summary = []

    for etf in etf_list:
        pred_path = os.path.join(output_root, etf, "prediction.csv")
        price_path = os.path.join(price_root, f"{etf}_features.csv")

        if not os.path.exists(pred_path) or not os.path.exists(price_path):
            print(f"[⚠️] {etf}: 데이터 누락, 건너뜀")
            continue

        result_df = simple_backtest(pred_path, price_path, etf)
        strategy_daily = result_df["Strategy"].pct_change().fillna(0).values
        bench_daily = result_df["Benchmark"].pct_change().fillna(0).values

        strat_metrics = compute_risk_metrics(strategy_daily)
        bench_metrics = compute_risk_metrics(bench_daily)

        result_summary.append({
            "ETF": etf,
            "Strategy Final Return": result_df["Strategy"].iloc[-1],
            "Benchmark Final Return": result_df["Benchmark"].iloc[-1],
            "Strategy Sharpe": strat_metrics["Sharpe"],
            "Strategy Volatility": strat_metrics["Volatility"],
            "Strategy MDD": strat_metrics["Max Drawdown"],
            "Benchmark Sharpe": bench_metrics["Sharpe"]
        })

    summary_df = pd.DataFrame(result_summary)
    return summary_df


def backtest_Model_all(etf_list, model_list, output_root="outputs", price_root="data/processed"):
    """
    ETF × 모델 조합을 모두 백테스트하고 전략 성과 및 리스크 지표 요약

    Parameters:
    - etf_list (list): 예측한 ETF 리스트 (예: ["XLK", "XLF"])
    - model_list (list): 모델 리스트 (예: ["Autoformer", "LSTM", "GRU"])
    - output_root (str): prediction 저장 루트 (outputs/{model}/{etf}_prediction.csv)
    - price_root (str): 가격 원본 저장 루트 (data/processed/{etf}_features.csv)

    Returns:
    - pd.DataFrame: 전체 결과 요약
    """
    result_summary = []

    for model in model_list:
        for etf in etf_list:
            pred_path = os.path.join(output_root, model, f"{etf}_prediction.csv")
            price_path = os.path.join(price_root, f"{etf}_features.csv")

            if not os.path.exists(pred_path) or not os.path.exists(price_path):
                print(f"[⚠️] {model}/{etf}: prediction or price data not found. Skipping.")
                continue

            result_df = simple_backtest(pred_path, price_path, etf=f"{etf} ({model})")
            strategy_daily = result_df["Strategy"].pct_change().fillna(0).values
            bench_daily = result_df["Benchmark"].pct_change().fillna(0).values

            strat_metrics = compute_risk_metrics(strategy_daily)
            bench_metrics = compute_risk_metrics(bench_daily)

            result_summary.append({
                "ETF": etf,
                "Model": model,
                "Strategy Final Return": result_df["Strategy"].iloc[-1],
                "Benchmark Final Return": result_df["Benchmark"].iloc[-1],
                "Sharpe": strat_metrics["Sharpe"],
                "Volatility": strat_metrics["Volatility"],
                "Max Drawdown": strat_metrics["Max Drawdown"],
                "Benchmark Sharpe": bench_metrics["Sharpe"]
            })

    summary_df = pd.DataFrame(result_summary)
    return summary_df