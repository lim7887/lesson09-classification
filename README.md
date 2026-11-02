import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/lim7887/data/main/stroke.csv"

BASE_DIR = Path(__file__).parent
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 기록 저장/불러오기 ----------
def load_record() -> dict:
    """내_기록/기록.json 을 읽어 온다. 없거나 깨졌으면 빈 기록을 돌려준다."""
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        data = {}
    data.setdefault("meanings", {})
    data.setdefault("source", "")
    return data


def save_record(meanings: dict | None = None, source: str | None = None) -> None:
    """바뀐 부분만 갱신해서 내_기록/기록.json 에 저장한다."""
    data = load_record()
    if meanings is not None:
        data["meanings"] = meanings
    if source is not None:
        data["source"] = source
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series: pd.Series) -> str:
    """열의 '값의 종류'를 짧은 글로 만든다."""
    values = series.dropna()
    unique = values.unique()
    if values.dtype == object or len(unique) <= 10:
        shown = sorted(unique.tolist(), key=lambda v: str(v))
        return ", ".join(str(v) for v in shown)
    return f"숫자 ({values.min():g} ~ {values.max():g})"


# ---------- 화면 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("첫 화면 · 이 데이터가 무엇인지 살펴봅니다")

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.markdown(
    "이 데이터는 사람마다 성별·나이·건강 상태·생활 습관 등을 적어 놓은 표이고, "
    "마지막 열 `stroke`는 뇌졸중 여부(1이면 뇌졸중, 0이면 아님)를 뜻합니다. "
    "이 실습실에서는 이 표를 바탕으로 뇌졸중을 예측해 봅니다."
)

# 1) 큰 숫자 카드 네 개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke가 1인 비율", f"{ratio:.1f}%")

# 2) 열 설명 표
st.subheader("열 설명 표")
st.write("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적은 뒤 **저장** 버튼을 누르세요.")

record = load_record()
saved_meanings = record["meanings"]

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [saved_meanings.get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

with st.form("meaning_form"):
    edited = st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻", width="medium"),
            "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        },
        key="meaning_editor",
    )
    if st.form_submit_button("우리말 뜻 저장", type="primary"):
        new_meanings = {
            row["열 이름"]: (row["우리말 뜻"] or "").strip()
            for _, row in edited.iterrows()
        }
        save_record(meanings=new_meanings)
        st.success("우리말 뜻을 저장했습니다.")

# 3) 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# 4) 데이터 출처
st.subheader("데이터 출처")
with st.form("source_form"):
    source_text = st.text_area(
        "교재에 나온 출처를 적어 주세요.",
        value=record["source"],
        height=120,
        label_visibility="collapsed",
        placeholder="교재에 나온 출처를 여기에 적어 주세요.",
    )
    if st.form_submit_button("출처 저장", type="primary"):
        save_record(source=source_text.strip())
        st.success("출처를 저장했습니다.")
