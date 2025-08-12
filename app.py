import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(page_title="착유량 예측기", page_icon="🐄")

# =========================
# 배경 이미지 (깃허브 raw URL) + 가독성 스타일
# =========================
BG_URL = "https://raw.githubusercontent.com/franklee0001/milking-yield-predict/main/background.png"
cache_buster = int(time.time() // 3600)  # 1시간 단위 캐시 무효화

#   
st.markdown(
    f"""
    <style>
    .stApp {{
      background-image:
        linear-gradient(rgba(0,0,0,0.16), rgba(0,0,0,0.16)),
        url("{BG_URL}?t={cache_buster}");
      background-size: cover;
      background-position: right center;
      background-repeat: no-repeat;
      background-color: #0f1116;
    }}

    /* 제목/본문 전반: 더 밝고 더 굵게 + 강한 그림자 */
    h1, h2, h3 {{
      color:#ffffff !important;
      font-weight: 900 !important;
      text-shadow:
        0 2px 3px rgba(0,0,0,.45),
        0 0 14px rgba(0,0,0,.25);
    }}
    .stMarkdown p, .stCaption, .stText {{
      color:#f8fafc !important;           /* 아주 밝은 흰 */
      font-weight: 700 !important;
      text-shadow: 0 1px 2px rgba(0,0,0,.35);
    }}
    label {{
      color:#ffffff !important;
      font-weight: 800 !important;
      text-shadow: 0 1px 2px rgba(0,0,0,.35);
    }}

    /* 입력 위젯 안 글자/placeholder도 또렷하게 */
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] input {{
      color:#ffffff !important;
      font-weight: 800 !important;
      text-shadow: 0 1px 2px rgba(0,0,0,.35);
    }}

    /* 버튼도 대비 높이기 */
    button[kind="primary"] {{
      font-weight: 800 !important;
      border-radius: 10px !important;
      box-shadow: 0 6px 18px rgba(0,0,0,.25);
    }}

    /* 메트릭 카드: 어두운 유리 → 노란 숫자가 튀도록 */
    .metric-card {{
      display:inline-block;
      background: rgba(0,0,0,0.55);
      -webkit-backdrop-filter: blur(6px);
      backdrop-filter: blur(6px);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 14px;
      padding: 12px 16px;
      margin-top: 10px;
      margin-bottom: 12px;
    }}
    [data-testid="stMetricValue"] {{
      color:#FFEB3B !important;           /* ▶ 노란색 강조 */
      font-weight: 1000 !important;
      font-size: 2.2rem !important;
      letter-spacing: .3px;
      text-shadow:
        0 2px 3px rgba(0,0,0,.55),
        0 0 18px rgba(0,0,0,.35);
    }}
    [data-testid="stMetricLabel"] {{
      color:#ffffff !important;
      font-weight: 900 !important;
      opacity:.98;
      text-shadow: 0 1px 2px rgba(0,0,0,.35);
    }}

    /* 등급 배지도 더 진하게 */
    .badge {{ color:#0f1116; font-weight: 900; }}
    .badge.lowfat {{ background:#90caf9; }}
    .badge.nofat  {{ background:#a5d6a7; }}
    .badge.normal {{ background:#e0e0e0; }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 모델 로드
# =========================
@st.cache_resource
def load_bundle():
    bundle = joblib.load("final_cb.pkl")   # 모델 번들: {"model": ..., "features": [...]}
    return bundle["model"], bundle["features"]

model, FEATURES = load_bundle()

# =========================
# 우유 등급 분류 함수
# =========================
def classify_milk(fat: float, protein: float):
    """
    Tableau 로직:
    IF fat>=3.8 AND protein>=3.2 -> 프리미엄
    ELSEIF fat<=2.0 AND fat>0.5  -> 저지방
    ELSEIF fat<=0.5              -> 무지방
    ELSE                         -> 일반
    """
    if fat >= 3.8 and protein >= 3.2:
        return "프리미엄 우유", "premium"
    elif fat <= 2.0 and fat > 0.5:
        return "저지방 우유", "lowfat"
    elif fat <= 0.5:
        return "무지방 우유", "nofat"
    else:
        return "일반 우유", "normal"

# =========================
# UI
# =========================
st.title("🐄 착유량 예측기 (CatBoost)")
st.caption("학습된 모델에 X값을 입력해 착유량(L)을 예측하고, 유지방/유단백에 따른 우유 등급을 안내합니다.")

col1, col2, col3 = st.columns(3)
with col1:
    온도 = st.number_input("온도(°C)", value=34.5, step=0.1)
    전도도 = st.number_input("전도도", value=7.0, step=0.1)
    착유회차 = st.number_input("착유회차", value=3, step=1, min_value=0)
with col2:
    혈액흐름 = st.number_input("혈액흐름", value=12.3, step=0.1)
    유지방 = st.number_input("유지방(%)", value=3.8, step=0.1, min_value=0.0)
    유단백 = st.number_input("유단백(%)", value=3.2, step=0.1, min_value=0.0)
with col3:
    공기흐름 = st.number_input("공기흐름", value=1.1, step=0.1)
    착유소요시간 = st.number_input("착유소요시간(분)", value=7.5, step=0.1)
    pfr_auto = st.toggle("PFR 자동 계산 (유단백/유지방)", value=True)

# PFR 자동/수동
if pfr_auto:
    PFR = (유단백 / 유지방) if 유지방 not in (0, None) and 유지방 != 0 else 0.0
else:
    PFR = st.number_input("PFR (유단백/유지방)", value=0.85, step=0.01)

# =========================
# 예측
# =========================
if st.button("예측하기"):
    row_full = {
        "온도": 온도,
        "전도도": 전도도,
        "착유회차": 착유회차,
        "혈액흐름": 혈액흐름,
        "유지방": 유지방,
        "유단백": 유단백,
        "공기흐름": 공기흐름,
        "PFR": PFR,
        "착유소요시간(분)": 착유소요시간,
    }
    # 학습 당시 FEATURES 순서에 맞추기
    row = {k: row_full[k] for k in FEATURES if k in row_full}
    X = pd.DataFrame([row], columns=FEATURES).astype(float)

    # 모델 예측
    y_pred = model.predict(X)
    y = float(y_pred[0])

    # 예측값 카드
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("예측된 착유량 (L)", f"{y:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 우유 등급 표시
    grade_text, tag = classify_milk(유지방, 유단백)
    # 등급 배지 색상 클래스 선택
    cls = {
        "premium": "",
        "lowfat": "lowfat",
        "nofat": "nofat",
        "normal": "normal"
    }[tag]
    st.markdown(
        f"<div style='margin-top:6px;font-weight:800;color:#fff;'>우유 등급: "
        f"<span class='badge {cls}'>{grade_text}</span></div>",
        unsafe_allow_html=True
    )

    # 입력값 확인
    with st.expander("입력값 확인"):
        st.write(X)
