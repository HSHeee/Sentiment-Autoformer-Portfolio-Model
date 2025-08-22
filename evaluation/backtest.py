import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import matplotlib.font_manager as fm
import platform

#plt.rcParams["font.family"] = "DejaVu Sans"

# 마이너스 부호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

def simple_backtest0(pred_path: str, price_path: str, etf: str, short_enabled: bool) -> pd.DataFrame:
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
        ret = merged["return"].fillna(0).values
    else:
        # 기존 방식 fallback
        y_true = pred_df["true"].values
        y_pred = pred_df["pred"].values
        start_idx = -len(y_true)
        ret = price_df["return"].fillna(0).values[start_idx:]

    signal = (y_pred > 0).astype(int)
    
    if short_enabled:
        short_signal = (y_pred < 0).astype(int)
        strategy_ret = signal * ret - short_signal * ret
    else:
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

def simple_backtest(
    pred_path: str,
    price_path: str,
    etf: str,
    short_enabled: bool,
    full_position: bool,          # 항상 롱/숏 유지
    threshold: float,              # 예: 0.0~0.2; 약한 신호는 유지
    hold_days: int,               # 예: 3 -> 진입 후 3영업일 유지 (고정보유)
    k_consecutive: int,           # 예: 2 -> 반대신호 2일 연속시 뒤집기 (히스테리시스)
    cost_bps: float,               # 매매 발생일에만 차감 (ex: 5bps = 0.0005)
    vol_target_annual: float,     # 예: 0.15 -> 변동성 타겟팅(선택)
    vol_lookback: int               # 변동성 추정 윈도우
) -> pd.DataFrame:
    """
    Autoformer 예측 기반 단순 전략 백테스트 (+ 풀포지션/홀드/거래비용/옵션)
    """

    # 1) 데이터 로드 & 머지
    pred_df = pd.read_csv(pred_path)
    px = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
    px = px[~px.index.duplicated(keep="first")]

    if "date" in pred_df.columns:
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        px_ = px.reset_index().rename(columns={"index": "date"})
        merged = pd.merge(pred_df, px_, on="date", how="inner")
        merged = merged.sort_values("date")
        y_pred = merged["pred"].to_numpy()
        # ret는 "다음날" 수익률이 적용되도록 한 칸 앞으로 당김(룩어헤드 방지)
        ret = merged["return"].fillna(0).shift(-1).to_numpy()
        dates = merged["date"].to_numpy()
    else:
        # fallback: 길이 맞춰 뒤쪽 정렬
        y_pred = pred_df["pred"].to_numpy()
        ret_series = px["return"].fillna(0)
        ret = ret_series.iloc[-len(y_pred):].shift(-1).to_numpy()
        dates = ret_series.index[-len(y_pred):].to_numpy()

    # 2) 초기 포지션 시그널
    # threshold로 약한 신호는 0 처리 (이후 로직에서 0은 '유지'로 바꿀 수 있음)
    raw_sign = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0)) #

    # 3) 풀포지션/유지 로직 구성
    # - full_position=True면 0이 나와도 직전 포지션 유지(초기 0은 +1로 시작하거나, 다음 비제로까지 대기)
    # - hold_days: 진입 후 N일 고정
    # - k_consecutive: 반대 신호가 k일 연속일 때만 뒤집기
    pos = np.zeros_like(raw_sign)

    if full_position:
        # 초기값 설정: 첫 비제로 시그널 찾기, 없으면 +1로 가정
        first_nonzero = np.flatnonzero(raw_sign)
        cur = raw_sign[first_nonzero[0]] if first_nonzero.size > 0 else 1
    else:
        cur = 0  # 캐시 허용 모드일 때는 0으로 시작

    # 상태 변수
    hold_counter = 0
    opposite_streak = 0

    for i, s in enumerate(raw_sign):
        proposed = cur  # 기본은 유지

        # 히스테리시스: 반대 시그널 연속 체크
        if k_consecutive is not None:
            if s == -cur and s != 0:
                opposite_streak += 1
            else:
                opposite_streak = 0

        # 고정 보유일수 우선 적용
        if hold_days is not None and hold_counter > 0:
            hold_counter -= 1
            proposed = cur  # 유지
        else:
            # 고정 보유 기간이 끝났을 때 또는 사용하지 않을 때만 업데이트 고려
            if s != 0:
                if k_consecutive is None:
                    proposed = s  # 즉시 반영
                else:
                    if s == -cur and opposite_streak >= k_consecutive:
                        proposed = s
                        opposite_streak = 0
                    elif s == cur:
                        proposed = cur  # 동방향이면 유지
                    # s==0은 위에서 처리 안 함(유지)
            else:
                if not full_position:
                    proposed = 0  # 캐시 허용이면 0 가능
                # full_position이면 유지

            # 포지션이 바뀌었으면 hold_days 리셋
            if hold_days is not None and proposed != cur:
                hold_counter = hold_days - 1  # 오늘 포함 N일 보유

        cur = proposed
        pos[i] = cur

    # full_position인데 끝까지 0만 있었다면 +1로 채우기(드문 케이스)
    if full_position and np.all(pos == 0):
        pos[:] = 1

    # 4) 거래비용 적용: 포지션 변화가 있는 날에만 비용 차감
    pos_shift = np.roll(pos, 1)
    pos_shift[0] = 0
    traded = (pos != pos_shift).astype(int)
    # 체결일 수익에 비용 반영(진입/청산/리버스 시 –cost)
    # bps를 소수로 변환
    cost = (cost_bps / 10000.0) * traded

    # 5) 전략 수익 계산 (ret는 t+1, pos는 t 기준 → 잘 정렬됨)
    # 숏 비활성화 시 롱만 취급하려면:
    if not short_enabled:
        eff_pos = np.where(pos > 0, 1, 0)
    else:
        eff_pos = pos  # -1, +1

    strat_ret = eff_pos * ret - cost

    # 6) 변동성 타겟팅(선택)
    # 일변동성 추정 → 목표 연변동성에 맞게 레버리지 스케일
    if vol_target_annual is not None:
        # 연->일 변환(거래일수 252 가정)
        target_daily_vol = vol_target_annual / np.sqrt(252.0)
        # 롤링 일변동성 추정(포지션 적용 전 원시 수익으로 추정해도 되고, 적용 후로 추정해도 됨)
        # 보수적으로는 전략 수익의 롤링 표준편차로 스케일링
        ser = pd.Series(strat_ret, index=pd.to_datetime(dates))
        rolling_vol = ser.rolling(vol_lookback).std().shift(1)  # 익일에 적용(룩어헤드 방지)
        leverage = (target_daily_vol / (rolling_vol.replace(0, np.nan))).clip(upper=5.0)  # 레버리지 상한
        leverage = leverage.fillna(0.0)
        strat_ret = (ser * leverage).to_numpy()

    # 7) 누적수익
    strategy_cum = np.cumprod(1.0 + np.nan_to_num(strat_ret, nan=0.0))
    bench_cum = np.cumprod(1.0 + np.nan_to_num(ret, nan=0.0))

    result = pd.DataFrame(
        {"Strategy": strategy_cum, "Benchmark": bench_cum, "Position": pos, "Traded": traded},
        index=pd.to_datetime(dates)
    )

    # 8) 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(result.index, result["Strategy"], label="Strategy")
    plt.plot(result.index, result["Benchmark"], label="Benchmark", linestyle="--")
    plt.title(f"{etf} Autoformer Returns")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    return result


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


def backtest_all0(etf_list, model_name="Autoformer", output_root="outputs", price_root="data/processed", short_enabled=False):
    """
    여러 ETF를 반복 백테스트하고 수익률/리스크 지표 비교

    Parameters:
    - etf_list: 예측한 ETF 티커 리스트
    - output_root: etf_prediction.csv 위치 기준
    - price_root: 실제 ETF 가격 (정규화된 features.csv 경로)

    Returns:
    - pd.DataFrame: ETF별 성과 요약 테이블
    """
    result_summary = []

    for etf in etf_list:
        pred_path = os.path.join(output_root, model_name, f"{etf}_prediction.csv")
        price_path = os.path.join(price_root, f"{etf}_features.csv")

        if not os.path.exists(pred_path) or not os.path.exists(price_path):
            print(f"[⚠️] {etf}: 데이터 누락, 건너뜀")
            continue

        result_df = simple_backtest(pred_path, price_path, etf, short_enabled)
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

def backtest_all(
    etf_list,
    model_name: str = "Autoformer",
    output_root: str = "outputs",
    price_root: str = "data/processed",
    short_enabled: bool = True,
    # 아래부터 simple_backtest로 패스되는 옵션들(원하면 생략 가능)
    full_position: bool = True,
    threshold: float = 0.00,
    hold_days: int | None = None,
    k_consecutive: int | None = None,
    cost_bps: float = 0.0,
    vol_target_annual: float | None = None,
    vol_lookback: int = 20,
    # 시각화 옵션
    plot_trades: bool = False,
    price_col_candidates=("Close","Adj Close","Adj_Close","close","adj_close","PX_LAST")
):
    """
    여러 ETF를 반복 백테스트하고 수익률/리스크 지표 비교

    Parameters:
    - etf_list: 예측한 ETF 티커 리스트
    - model_name/output_root: {output_root}/{model_name}/{etf}_prediction.csv
    - price_root: {price_root}/{etf}_features.csv (return/가격 포함)
    - short_enabled: 숏 사용 여부
    - full_position/threshold/hold_days/k_consecutive/cost_bps/vol_target_annual/vol_lookback:
        simple_backtest 옵션 그대로 전달
    - plot_trades: 각 종목별 가격 위에 매수(빨강)/매도(파랑) 점 찍어 표시

    Returns:
    - pd.DataFrame: ETF별 성과 요약 테이블
    """
    result_summary = []

    for etf in etf_list:
        pred_path = os.path.join(output_root, model_name, f"{etf}_prediction.csv")
        price_path = os.path.join(price_root, f"{etf}_features.csv")

        if not os.path.exists(pred_path) or not os.path.exists(price_path):
            print(f"[⚠️] {etf}: 데이터 누락, 건너뜀")
            continue

        # --- 전략 백테스트 실행 ---
        result_df = simple_backtest(
            pred_path=pred_path,
            price_path=price_path,
            etf=etf,
            short_enabled=short_enabled,
            full_position=full_position,
            threshold=threshold,
            hold_days=hold_days,
            k_consecutive=k_consecutive,
            cost_bps=cost_bps,
            vol_target_annual=vol_target_annual,
            vol_lookback=vol_lookback
        )

        # --- 요약 지표 계산 ---
        strategy_daily = pd.Series(result_df["Strategy"]).pct_change().fillna(0).values
        bench_daily    = pd.Series(result_df["Benchmark"]).pct_change().fillna(0).values

        strat_metrics = compute_risk_metrics(strategy_daily)
        bench_metrics = compute_risk_metrics(bench_daily)

        result_summary.append({
            "ETF": etf,
            "Strategy Final Return": result_df["Strategy"].iloc[-1],
            "Benchmark Final Return": result_df["Benchmark"].iloc[-1],
            "Strategy Sharpe": strat_metrics.get("Sharpe", np.nan),
            "Strategy Volatility": strat_metrics.get("Volatility", np.nan),
            "Strategy MDD": strat_metrics.get("Max Drawdown", np.nan),
            "Benchmark Sharpe": bench_metrics.get("Sharpe", np.nan),
        })

        # --- (옵션) 트레이드 점 시각화 ---
        if plot_trades:
            px = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
            # 인덱스 정렬/교집합은 함수 내부에서 다시 처리되지만, 아래처럼 넘기면 충분
            plot_price_with_trades_dualaxis(
                price_df=px,
                position=result_df["Position"],
                cumret_df=result_df[["Strategy","Benchmark"]],
                title=f"{etf} - Price/Index + Trades & CumReturn",
                base_value=100.0,
                save_path=f"outputs/plots/{etf}_dualaxis.png"   # headless 서버면 저장 권장
            )

    summary_df = pd.DataFrame(result_summary)
    return summary_df

def backtest_Model_all(etf_list, model_list, output_root="outputs", price_root="data/processed", short_enabled = False):

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

            result_df = simple_backtest(pred_path, price_path, short_enabled, etf=f"{etf} ({model})")
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

def plot_price_with_trades_dualaxis(
    price_df: pd.DataFrame,
    position: pd.Series | np.ndarray,
    cumret_df: pd.DataFrame,   # ["Strategy","Benchmark"]
    date_col: str | None = None,
    price_col_candidates=("Close","Adj Close","Adj_Close","close","adj_close","PX_LAST"),
    title: str = "Price + Trades & Cumulative Returns",
    base_value: float = 100.0,
    save_path: str | None = None
):
    """
    한 그래프에:
      - 좌측 y축: 가격/인덱스 + 매수/매도 점
      - 우측 y축: 누적수익률 (Strategy / Benchmark)
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    # 날짜 정리
    if date_col is not None:
        price_df = price_df.copy()
        price_df[date_col] = pd.to_datetime(price_df[date_col])
        price_df = price_df.sort_values(date_col).set_index(date_col)
    else:
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("price_df는 DatetimeIndex이거나 date_col을 지정해야 합니다.")
        price_df = price_df.sort_index()

    cumret_df = cumret_df.copy()
    cumret_df.index = pd.to_datetime(cumret_df.index)
    cumret_df = cumret_df.sort_index()

    # 포지션 정렬
    if isinstance(position, np.ndarray):
        pos = pd.Series(position, index=price_df.index).astype(float)
    else:
        pos = position.reindex(price_df.index).ffill().fillna(0).astype(float)

    # 가격 시리즈 추출 or return 기반 재구성
    price_col = None
    for c in price_col_candidates:
        if c in price_df.columns:
            price_col = c
            break
    if price_col is not None:
        price_series = price_df[price_col].astype(float)
        y_label_left = "Price"
    else:
        if "return" not in price_df.columns:
            raise ValueError("가격 컬럼도 'return'도 없음.")
        ret = price_df["return"].fillna(0).astype(float)
        price_series = (1.0 + ret).cumprod() * base_value
        y_label_left = f"Index (base={base_value:.0f})"

    # 공통 인덱스
    common_idx = price_series.index.intersection(cumret_df.index)
    price_series = price_series.loc[common_idx]
    pos = pos.loc[common_idx]
    cumret_df = cumret_df.loc[common_idx]

    # 매수/매도 마스크
    pos_shift = pos.shift(1).fillna(0)
    buy_mask  = (pos > pos_shift)
    sell_mask = (pos < pos_shift)

    buy_dates  = price_series.index[buy_mask]
    sell_dates = price_series.index[sell_mask]
    buy_prices  = price_series.loc[buy_dates]
    sell_prices = price_series.loc[sell_dates]

    # 그림
    fig, ax1 = plt.subplots(figsize=(11, 5))

    # 좌측 y축: 가격 + 매수/매도 점
    ax1.plot(price_series.index, price_series.values, linewidth=1.3, label="Price/Index", color="black")
    ax1.scatter(buy_dates,  buy_prices,  s=28, marker='o', color='red',  label='Buy',  zorder=3)
    ax1.scatter(sell_dates, sell_prices, s=28, marker='o', color='blue', label='Sell', zorder=3)
    ax1.set_ylabel(y_label_left)
    ax1.grid(True, alpha=0.3)

    # 우측 y축: 누적 수익률
    ax2 = ax1.twinx()
    ax2.plot(cumret_df.index, cumret_df["Strategy"], label="Strategy", color="green")
    ax2.plot(cumret_df.index, cumret_df["Benchmark"], label="Benchmark", linestyle="--", color="orange")
    ax2.set_ylabel("Cumulative Return")

    # 범례 합치기
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # 날짜 포맷
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            plt.show()
        else:
            safe_title = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in title)
            auto_path = f"outputs/plots/{safe_title}.png"
            os.makedirs(os.path.dirname(auto_path), exist_ok=True)
            plt.savefig(auto_path, dpi=150)
            plt.close()
            print(f"[i] headless 환경 감지: '{auto_path}' 저장 완료")

def plot_close_and_cumret_dualaxis(
    price_df: pd.DataFrame,
    cumret_df: pd.DataFrame,               # columns: ["Strategy","Benchmark"]
    position: pd.Series | np.ndarray,      # -1/0/+1 (가격 인덱스에 맞춤)
    date_col: str | None = None,
    price_col_candidates=("close","Close","Adj Close","Adj_Close","PX_LAST","adj_close"),
    title: str = "Close + Trades & Cumulative Returns",
    save_path: str | None = None
):
    """
    한 그래프에:
      - 좌측 y축: 'close'(또는 후보 컬럼) 가격 + 매수(빨강)/매도(파랑) 점
      - 우측 y축: 누적수익률 (Strategy / Benchmark)

    * price_df: DatetimeIndex 권장. 없으면 date_col 지정.
    * cumret_df: result_df[["Strategy","Benchmark"]] (index: 날짜)
    * position: result_df["Position"] (Series 추천; ndarray도 가능)
    """

    # 1) 날짜 정리
    if date_col is not None:
        price_df = price_df.copy()
        price_df[date_col] = pd.to_datetime(price_df[date_col])
        price_df = price_df.sort_values(date_col).set_index(date_col)
    else:
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("price_df는 DatetimeIndex이거나 date_col을 지정해야 합니다.")
        price_df = price_df.sort_index()

    cumret_df = cumret_df.copy()
    cumret_df.index = pd.to_datetime(cumret_df.index)
    cumret_df = cumret_df.sort_index()

    # 2) 가격 컬럼 선택 (close 우선)
    price_col = None
    for c in price_col_candidates:
        if c in price_df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError(f"'close' 가격 컬럼을 찾지 못했습니다. candidates={price_col_candidates}")

    price_series = price_df[price_col].astype(float)

    # 3) 공통 인덱스 정렬
    common_idx = price_series.index.intersection(cumret_df.index)
    if isinstance(position, np.ndarray):
        if len(position) != len(price_df):
            raise ValueError(f"position 길이({len(position)})와 price_df 길이({len(price_df)})가 다릅니다.")
        pos = pd.Series(position, index=price_df.index).astype(float).loc[common_idx]
    else:
        pos = pd.Series(position).copy()
        pos.index = pd.to_datetime(pos.index)
        pos = pos.sort_index().reindex(common_idx).ffill().fillna(0).astype(float)

    price_series = price_series.loc[common_idx]
    cumret_df = cumret_df.loc[common_idx]

    # 4) 매수/매도 마스크 (포지션 변화 기반)
    pos_shift = pos.shift(1).fillna(0)
    buy_mask  = (pos > pos_shift)
    sell_mask = (pos < pos_shift)

    buy_dates  = price_series.index[buy_mask];  buy_prices  = price_series.loc[buy_dates]
    sell_dates = price_series.index[sell_mask]; sell_prices = price_series.loc[sell_dates]

    # 5) 플롯: 좌측 가격 + 점, 우측 누적수익률
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(price_series.index, price_series.values, linewidth=1.3, label="Close")
    ax1.scatter(buy_dates,  buy_prices,  s=28, marker='o', color='red',  label='Buy',  zorder=3)
    ax1.scatter(sell_dates, sell_prices, s=28, marker='o', color='blue', label='Sell', zorder=3)
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(cumret_df.index, cumret_df["Strategy"],  label="Strategy",  linestyle="-")
    ax2.plot(cumret_df.index, cumret_df["Benchmark"], label="Benchmark", linestyle="--")
    ax2.set_ylabel("Cumulative Return")

    # 범례 통합
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # 날짜 축 포맷
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.suptitle(title)
    plt.tight_layout()

    # 6) 저장/표시 (headless 자동 저장)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            plt.show()
        else:
            safe = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in title)
            auto_path = f"outputs/plots/{safe}.png"
            os.makedirs(os.path.dirname(auto_path), exist_ok=True)
            plt.savefig(auto_path, dpi=150)
            plt.close()
            print(f"[i] headless 환경: '{auto_path}' 저장")

