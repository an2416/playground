import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
from typing import List, Dict, Union
from datetime import timedelta, date

cache_path = os.path.join(os.path.expanduser("~"), ".matplotlib", "fontlist-v390.json")
if os.path.exists(cache_path):
    os.remove(cache_path)

# 설정
np.random.seed(42)
today = pd.to_datetime(date.today())

# 연락자별 랜덤 연락 기록 생성 (오늘 기준 과거 날짜 생성)
TOTAL_NUM_OF_CONTACT_DAYS = 365
MIN_CONTACT_COUNT = 10
MAX_CONTACT_COUNT = 100

contacts_logs = {
    name: sorted(
        today - pd.to_timedelta(np.random.choice(range(TOTAL_NUM_OF_CONTACT_DAYS), size=np.random.randint(MIN_CONTACT_COUNT, MAX_CONTACT_COUNT), replace=False), unit='D'))
    for name in ["Alice", "Bob", "Charlie", "David"]
}

# 한글 폰트 설정 (맑은 고딕)
if sys.platform == "darwin":
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    font_path = 'fonts/NanumGothic.ttf'
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()
    print(f"Loaded font name: {fontprop.get_name()}")
plt.rcParams['axes.unicode_minus'] = False

font_list = [f.name for f in fm.fontManager.ttflist]
print("설치된 폰트 목록:", font_list)



# alpha 자동 계산 함수 (hybrid)
def compute_dynamic_alpha(intervals: List[int]) -> float:
    variance = np.var(intervals)

    N_base = 3
    N_ratio = 0.2
    N = max(N_base, int(N_ratio * len(intervals)))
    recent_intervals = intervals[-N:]

    recent_change = np.std(recent_intervals) / np.mean(recent_intervals) if np.mean(recent_intervals) != 0 else 0

    variance_low = 50
    variance_high = 500

    change_low = 0.05
    change_high = 0.5

    alpha_var = np.clip((variance - variance_low) / (variance_high - variance_low), 0, 1)
    alpha_change = np.clip((recent_change - change_low) / (change_high - change_low), 0, 1)

    alpha = 0.1 + 0.3 * max(alpha_var, alpha_change)

    return alpha


# Confidence Score 계산 함수
def compute_confidence(intervals: np.ndarray, ema_history: List[float], gamma: float = 0.9) -> float:
    ema_history = np.array(ema_history)
    relative_errors = np.abs(intervals - ema_history) / (ema_history + 1e-6)
    weights = gamma ** np.arange(len(relative_errors))[::-1]
    weighted_error = np.sum(relative_errors * weights) / np.sum(weights)
    confidence = max(0, 1 - weighted_error)
    return confidence


# 리마인더 시스템 함수 정의
def robust_contact_reminder(
        contact_name: str,
        contact_history: List[pd.Timestamp],
        z_threshold: float = 1.0,
        sudden_change_threshold: float = 0.2
) -> Dict[str, Union[str, float, List[float]]]:
    contact_history = sorted(contact_history)

    if len(contact_history) < 2:
        return {
            "Name": contact_name,
            "Last Contact": contact_history[-1].date(),
            "Next Expected": "N/A",
            "Mean Interval": "N/A",
            "EMA Interval": "N/A",
            "z-Score": "N/A",
            "Confidence": "N/A",
            "Confidence_raw": "N/A",
            "Status": f"\U0001f4c9 데이터 부족: {len(contact_history)}개 이력",
            "EMA History": [],
            "Alpha": "N/A"
        }

    intervals = np.diff(contact_history).astype('timedelta64[D]').astype(int)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    variance = np.var(intervals)
    last_contact = contact_history[-1]
    days_since_last = (today - last_contact).days

    alpha = compute_dynamic_alpha(intervals)

    ema_interval = intervals[0]
    ema_history = [ema_interval]

    for i in range(1, len(intervals)):
        change_ratio = intervals[i] / ema_interval if ema_interval != 0 else 1

        if change_ratio < sudden_change_threshold:
            weight = 0.1
        else:
            weight = 1.0

        recent_error = abs(intervals[i] - ema_interval) / (ema_interval + 1e-6)
        if recent_error > 0.5:
            local_alpha = min(alpha * 1.5, 0.5)
        elif recent_error < 0.1:
            local_alpha = max(alpha * 0.7, 0.05)
        else:
            local_alpha = alpha

        ema_interval = local_alpha * weight * intervals[i] + (1 - local_alpha) * ema_interval
        ema_history.append(ema_interval)

    # Confidence Score 계산
    confidence = compute_confidence(intervals, ema_history)

    # Data Sufficiency Factor 적용
    full_data_count = 8
    interval_count = len(intervals)

    if interval_count >= full_data_count:
        sufficiency_factor = 1.0
    else:
        sufficiency_factor = interval_count / full_data_count
        sufficiency_factor = max(sufficiency_factor, 0.3)

    confidence_final = confidence * sufficiency_factor

    dynamic_min_points = max(3, int(30 / mean_interval))
    relative_variance_threshold = (mean_interval * 1.5) ** 2

    if len(contact_history) < dynamic_min_points:
        return {
            "Name": contact_name,
            "Last Contact": last_contact.date(),
            "Next Expected": "N/A",
            "Mean Interval": round(mean_interval, 2),
            "EMA Interval": round(ema_interval, 2),
            "z-Score": "N/A",
            "Confidence": round(confidence_final, 3),
            "Confidence_raw": round(confidence, 3),
            "Status": f"\U0001f4c9 이력 부족 (동적 기준: {dynamic_min_points}개)",
            "EMA History": ema_history,
            "Alpha": round(alpha, 3)
        }

    if variance > relative_variance_threshold or std_interval == 0:
        return {
            "Name": contact_name,
            "Last Contact": last_contact.date(),
            "Next Expected": "N/A",
            "Mean Interval": round(mean_interval, 2),
            "EMA Interval": round(ema_interval, 2),
            "z-Score": "N/A",
            "Confidence": round(confidence_final, 3),
            "Confidence_raw": round(confidence, 3),
            "Status": f"❓ 패턴 불분명: 분산 {variance:.2f} (허용치 {relative_variance_threshold:.2f})",
            "EMA History": ema_history,
            "Alpha": round(alpha, 3)
        }

    z_score = (days_since_last - ema_interval) / std_interval
    if z_score > z_threshold:
        status = f"\U0001f514 z-score {z_score:.2f}: 연락 필요!"
    elif days_since_last >= ema_interval:
        status = f"\U0001f552 EMA 주기 도달 (z-score {z_score:.2f})"
    else:
        status = f"⏳ 아직 {abs(days_since_last - ema_interval):.0f}일 여유 있음 (z-score {z_score:.2f})"

    return {
        "Name": contact_name,
        "Last Contact": last_contact.date(),
        "Next Expected": (last_contact + timedelta(days=int(ema_interval))).date(),
        "Mean Interval": round(mean_interval, 2),
        "EMA Interval": round(ema_interval, 2),
        "z-Score": round(z_score, 2),
        "Confidence": round(confidence_final, 3),
        "Confidence_raw": round(confidence, 3),
        "Status": status,
        "EMA History": ema_history,
        "Alpha": round(alpha, 3)
    }


# Streamlit 앱 시작
st.title(f"📅 연락 리마인더 시스템 (EMA + 특수 이벤트 고려 + Adaptive Alpha + Confidence 개선) {font_list}")

# 사용자 슬라이더로 sudden_change_threshold 설정 가능
sudden_change_threshold = st.sidebar.slider("급격한 변화 판정 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# 사용자 입력 데이터 테스트 (항상 최상단에 표시)
st.sidebar.header("📥 사용자 입력 데이터 테스트")
user_input = st.sidebar.text_area(
    "연락 날짜 입력 (YYYY-MM-DD 형식, 줄바꿈 구분)",
    "2025-01-01\n2025-01-15\n2025-02-10\n2025-03-01"
)

if st.sidebar.button("결과 보기"):
    try:
        user_dates = [
            pd.to_datetime(line.strip())
            for line in user_input.strip().split("\n")
            if line.strip() != ""
        ]
        user_dates = sorted(user_dates)

        user_result = robust_contact_reminder("사용자 입력", user_dates, sudden_change_threshold=sudden_change_threshold)

        st.subheader("📋 사용자 입력 결과")
        df_user_result = pd.DataFrame([user_result])
        display_columns = [
            "Name", "Last Contact", "Next Expected", "Mean Interval",
            "EMA Interval", "z-Score", "Confidence", "Confidence_raw", "Status"
        ]
        st.dataframe(df_user_result[display_columns])

        st.markdown("### 사용자 입력 연락 날짜 도표")
        dates_numeric = [(d - today).days for d in user_dates]
        fig1, ax1 = plt.subplots(figsize=(8, 1))
        ax1.scatter(dates_numeric, [1] * len(dates_numeric), marker='o', color='green')
        ax1.set_yticks([])
        ax1.set_xlabel("오늘로부터 경과일")
        ax1.set_title("사용자 입력 연락 날짜")
        st.pyplot(fig1)

        if len(user_dates) > 1:
            intervals = np.diff(user_dates).astype('timedelta64[D]').astype(int)
            ema_history = user_result['EMA History']

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(range(1, len(intervals) + 1), intervals, marker='o', linestyle='-', label='Interval days')
            ax2.plot(range(1, len(ema_history) + 1), ema_history, marker='x', linestyle='--', color='red', label='EMA')
            ax2.set_xlabel("연락 순서")
            ax2.set_ylabel("Interval days")
            ax2.set_title("사용자 입력 연락 날짜 간 interval days + EMA")
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"입력 데이터 파싱 중 오류 발생: {e}")

# 기존 연락자 리마인더 결과
reminder_results = []
reminder_results_raw = {}
for name, history in contacts_logs.items():
    result = robust_contact_reminder(name, history, sudden_change_threshold=sudden_change_threshold)
    reminder_results.append(result)
    reminder_results_raw[name] = result

st.subheader("연락자별 리마인더 상태")
display_columns = [
    "Name", "Last Contact", "Next Expected", "Mean Interval",
    "EMA Interval", "z-Score", "Confidence", "Confidence_raw", "Status"
]
df_reminder_display = pd.DataFrame(reminder_results)[display_columns]
st.dataframe(df_reminder_display)

# 연락자별 연락 날짜 및 interval 도표
st.subheader("📅 연락자별 연락 날짜 및 간격 분석")
for name, history in contacts_logs.items():
    st.markdown(f"### {name}와 연락한 날짜 도표")

    dates_df = pd.DataFrame({
        '연락 날짜': [d.date() for d in history]
    })
    st.dataframe(dates_df)

    fig1, ax1 = plt.subplots(figsize=(8, 1))
    dates_numeric = [(d - today).days for d in history]
    ax1.scatter(dates_numeric, [1] * len(dates_numeric), marker='o', color='blue')
    ax1.set_yticks([])
    ax1.set_xlabel("오늘로부터 경과일")
    ax1.set_title(f"{name} 연락한 날짜")
    st.pyplot(fig1)

    if len(history) > 1:
        intervals = np.diff(history).astype('timedelta64[D]').astype(int)
        ema_history = reminder_results_raw[name]['EMA History']

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(range(1, len(intervals) + 1), intervals, marker='o', linestyle='-', label='Interval days')
        ax2.plot(range(1, len(ema_history) + 1), ema_history, marker='x', linestyle='--', color='red', label='EMA')
        ax2.set_xlabel("연락 순서")
        ax2.set_ylabel("Interval days")
        ax2.set_title(f"{name} 연락 날짜 간 interval days + EMA")
        ax2.legend()
        st.pyplot(fig2)

# 연락 주기 분포 히스토그램
st.subheader("연락 주기 분포 히스토그램")
fig, ax = plt.subplots(figsize=(10, 4))
for name, history in contacts_logs.items():
    if len(history) > 1:
        intervals = np.diff(history).astype('timedelta64[D]').astype(int)
        ax.hist(intervals, bins=10, alpha=0.5, label=name)

ax.set_xlabel("연락 간격 (일)")
ax.set_ylabel("빈도")
ax.set_title("연락자별 연락 주기 분포")
ax.legend()
ax.grid(True)
st.pyplot(fig)
