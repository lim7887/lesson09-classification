import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "탐색"
PAGE_ICON = "🔍"
DATA_URL = "https://raw.githubusercontent.com/lim7887/data/main/stroke.csv"

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}", page_icon=PAGE_ICON, layout="wide"
)

STROKE_LABEL = {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
GROUP_COLORS = {"뇌졸중 없음": "#4C78A8", "뇌졸중 있음": "#E45756"}


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_by(df: pd.DataFrame, column: str, names: dict) -> pd.DataFrame:
    """column 값(0/1)별 사람 수, 뇌졸중 사람 수, 뇌졸중 비율(%)을 계산한다."""
    grouped = df.groupby(column)["stroke"].agg(사람수="count", 뇌졸중사람수="sum")
    grouped["뇌졸중 비율(%)"] = grouped["뇌졸중사람수"] / grouped["사람수"] * 100
    grouped = grouped.reset_index()
    grouped["구분"] = grouped[column].map(names)
    return grouped


def rate_bar(df: pd.DataFrame, column: str, names: dict, title: str):
    rate = stroke_rate_by(df, column, names)
    fig = px.bar(
        rate,
        x="구분",
        y="뇌졸중 비율(%)",
        text=rate["뇌졸중 비율(%)"].map(lambda v: f"{v:.1f}%"),
        title=title,
        color="구분",
        color_discrete_sequence=["#4C78A8", "#E45756"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="뇌졸중 비율(%)",
        yaxis_range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.25, 1)],
    )
    return fig


st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · 데이터를 그림과 표로 살펴봅니다")

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map(STROKE_LABEL)

# 1) 나이와 평균 혈당의 분포
st.subheader("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)
with left:
    fig = px.histogram(
        df, x="age", nbins=30, title="나이 분포", labels={"age": "나이(세)"}
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=40,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# 2) 뇌졸중 여부에 따른 나이·평균 혈당 비교
st.subheader("2. 뇌졸중 여부에 따른 비교")
order = ["뇌졸중 없음", "뇌졸중 있음"]
left, right = st.columns(2)
with left:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": order},
        color_discrete_map=GROUP_COLORS,
        title="나이 비교",
        labels={"age": "나이(세)"},
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": order},
        color_discrete_map=GROUP_COLORS,
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

means = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(order)
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당의 평균"})
    .round(2)
    .reset_index()
)
st.write("두 그룹의 평균값")
st.dataframe(means, hide_index=True, use_container_width=True)

# 3) 고혈압·심장병과 뇌졸중 비율
st.subheader("3. 고혈압·심장병에 따른 뇌졸중 비율")
left, right = st.columns(2)
with left:
    st.plotly_chart(
        rate_bar(df, "hypertension", {0: "고혈압 없음", 1: "고혈압 있음"}, "고혈압"),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        rate_bar(df, "heart_disease", {0: "심장병 없음", 1: "심장병 있음"}, "심장병"),
        use_container_width=True,
    )

# 4) bmi가 빈 사람들
st.subheader("4. 체질량지수(bmi)가 빈 사람들")
bmi_empty = df[df["bmi"].isna()]
rows = []
for name, part in [("bmi가 빈 사람", bmi_empty), ("전체", df)]:
    n = len(part)
    n_stroke = int((part["stroke"] == 1).sum())
    rows.append(
        {
            "구분": name,
            "사람 수": n,
            "뇌졸중 사람 수": n_stroke,
            "뇌졸중 비율(%)": round(n_stroke / n * 100, 2) if n else 0.0,
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# 5) 흡연 상태별 사람 수
st.subheader("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True, use_container_width=True)
