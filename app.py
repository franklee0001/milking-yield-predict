import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(page_title="착유량 예측기", page_icon="🐄")

# 🎨 페이지 배경 스타일 적용
BG_URL = "https://raw.githubusercontent.com/franklee0001/milking-yield-predict/main/background.png"
cache_buster = int(time.time() // 3600)  # 캐시 무효화(1시간 단위)

st.markdown(
    f"""
    <style>
    /* 전체 글씨 색상과 굵기 */
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)),
            url("{BG_URL}?t={cache_buster}");
        background-size: cover;
        background-position: right center;
        background-repeat: no-repeat;
        background-color: #0f1116;
        color: white !important;
        font-weight: 600 !important;
    }}

    /* metric 카드 글씨 굵게 & 배경 반투명 */
    [data-testid="stMetricValue"] {{
        color: #ffeb3b !important;  /* 노란색 강조 */
        font-size: 2rem !important;
        font-weight: bold !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: white !important;
        font-weight: bold !important;
    }}
    .stMetric {{
        background-color: rgba(0,0,0,0.5);
        padding: 10px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_bundle():
    bundle = joblib.load("final_cb.pkl")
    return bundle["model"], bundle["features"]

model, FEATURES = load_bundle()

st.title("🐄 착유량 예측기 (CatBoost)")
st.caption("학습된 모델에 X값을 입력해 착유량(L)을 예측합니다.")

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

if pfr_auto:
    PFR = (유단백 / 유지방) if 유지방 not in (0, None) and 유지방 != 0 else 0.0
else:
    PFR = st.number_input("PFR (유단백/유지방)", value=0.85, step=0.01)

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
    row = {k: row_full[k] for k in FEATURES if k in row_full}
    X = pd.DataFrame([row], columns=FEATURES).astype(float)

    y_pred = model.predict(X)
    st.metric("예측된 착유량 (L)", f"{float(y_pred[0]):.2f}")
    with st.expander("입력값 확인"):
        st.write(X)
