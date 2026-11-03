import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "채점표"
PAGE_ICON = "📝"
DATA_URL = "https://raw.githubusercontent.com/lim7887/data/main/stroke.csv"

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}", page_icon=PAGE_ICON, layout="wide"
)

# '분류 모델' 페이지에서 처음 골라져 있는 입력 속성 넷(체질량지수 제외)을 그대로 쓴다.
FEATURES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
COLS = list(FEATURES)

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "다수 클래스 기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

NA_TEXT = "계산할 수 없음"
MAX_ROWS = 20

CELL_COLORS = {"total": "#EBEBEB", "right": "#CFE3F5", "wrong": "#F8D5D5"}


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def count_cells(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """뇌졸중(1)을 양성으로 두고 네 칸의 사람 수를 센다."""
    return {
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
    }


def metrics_from(c: dict) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 분모가 0이면 None."""
    total = c["TP"] + c["FN"] + c["FP"] + c["TN"]
    actual_pos = c["TP"] + c["FN"]  # 실제 뇌졸중인 사람 수
    pred_pos = c["TP"] + c["FP"]  # 뇌졸중이라 예측한 사람 수
    accuracy = (c["TP"] + c["TN"]) / total if total else None
    recall = c["TP"] / actual_pos if actual_pos else None
    precision = c["TP"] / pred_pos if pred_pos else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def pct(v) -> str:
    return NA_TEXT if v is None else f"{v * 100:.2f}%"


def confusion_figure(c: dict) -> go.Figure:
    """행은 실제, 열은 예측인 2×2 혼동행렬(합계 포함)을 그린다."""
    n_actual_pos = c["TP"] + c["FN"]
    n_actual_neg = c["FP"] + c["TN"]
    n_pred_pos = c["TP"] + c["FP"]
    n_pred_neg = c["FN"] + c["TN"]
    total = n_actual_pos + n_actual_neg

    x_labels = ["뇌졸중으로 예측", "아님으로 예측", "합계"]
    y_labels = ["실제 뇌졸중", "실제 아님", "합계"]
    # 0: 합계 칸, 1: 맞힌 칸, 2: 틀린 칸
    z = [[1, 2, 0], [2, 1, 0], [0, 0, 0]]
    third = 1 / 3
    colorscale = [
        [0, CELL_COLORS["total"]],
        [third, CELL_COLORS["total"]],
        [third, CELL_COLORS["right"]],
        [2 * third, CELL_COLORS["right"]],
        [2 * third, CELL_COLORS["wrong"]],
        [1, CELL_COLORS["wrong"]],
    ]
    fig = go.Figure(
        go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=z,
            zmin=0,
            zmax=2,
            colorscale=colorscale,
            showscale=False,
            xgap=3,
            ygap=3,
            hoverinfo="skip",
        )
    )
    cells = [
        ("실제 뇌졸중", "뇌졸중으로 예측", "TP", "뇌졸중을 맞힘", c["TP"]),
        ("실제 뇌졸중", "아님으로 예측", "FN", "놓친 환자", c["FN"]),
        ("실제 아님", "뇌졸중으로 예측", "FP", "헛짚은 사람", c["FP"]),
        ("실제 아님", "아님으로 예측", "TN", "아님을 맞힘", c["TN"]),
    ]
    for y, x, abbr, desc, value in cells:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{abbr}</b> ({desc})<br><span style='font-size:22px'><b>{value:,}명</b></span>",
            showarrow=False,
            font=dict(color="#222222", size=13),
        )
    totals = [
        ("실제 뇌졸중", "합계", n_actual_pos),
        ("실제 아님", "합계", n_actual_neg),
        ("합계", "뇌졸중으로 예측", n_pred_pos),
        ("합계", "아님으로 예측", n_pred_neg),
        ("합계", "합계", total),
    ]
    for y, x, value in totals:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{value:,}명</b>",
            showarrow=False,
            font=dict(color="#222222", size=15),
        )
    fig.update_xaxes(side="top", title="예측")
    fig.update_yaxes(autorange="reversed", title="실제")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=70, b=10))
    return fig


def person_list(rows: pd.DataFrame) -> pd.DataFrame:
    """번호·나이·평균 혈당·고혈압 여부·심장병 여부를 나이 많은 순으로 정렬한다."""
    rows = rows.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": rows["id"].to_numpy(),
            "나이": rows["age"].to_numpy(),
            "평균 혈당": rows["avg_glucose_level"].round(2).to_numpy(),
            "고혈압 여부": np.where(rows["hypertension"] == 1, "있음", "없음"),
            "심장병 여부": np.where(rows["heart_disease"] == 1, "있음", "없음"),
        }
    )


def show_list(title_text: str, rows: pd.DataFrame) -> None:
    n = len(rows)
    shown = person_list(rows).head(MAX_ROWS)
    if n > MAX_ROWS:
        st.markdown(f"**{title_text} — 전체 {n:,}명 (나이 많은 순 앞 {MAX_ROWS}명만 표시)**")
    else:
        st.markdown(f"**{title_text} — 전체 {n:,}명**")
    st.dataframe(shown, hide_index=True, use_container_width=True)


st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · 두 모델이 테스트 데이터에서 어디를 맞히고 어디를 틀렸는지 채점합니다")

try:
    raw = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ---------- '분류 모델' 페이지와 같은 방법으로 나누고 학습 ----------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3  # 열 명씩 묶어 앞 세 명이 테스트용
X_train = df.loc[~is_test, COLS]
y_train = df.loc[~is_test, "stroke"]
test_df = df.loc[is_test].reset_index(drop=True)
X_test = test_df[COLS]
y_test = test_df["stroke"].to_numpy()

logi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logi.fit(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

models = {LOGI_NAME: logi, TREE_NAME: tree, BASE_NAME: base}
predictions = {name: m.predict(X_test) for name, m in models.items()}
cells = {name: count_cells(y_test, p) for name, p in predictions.items()}

st.caption(
    "'분류 모델' 페이지와 똑같이 번호 순으로 정렬해 열 명씩 묶고 앞 세 명을 테스트용으로 고정했으며, "
    f"입력 속성은 처음 골라져 있는 {', '.join(FEATURES.values())} 넷입니다. "
    "뇌졸중을 양성으로 둡니다."
)

# ---------- 혼동행렬 ----------
st.subheader("1. 혼동행렬")
n_test = len(test_df)
n_test_pos = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 실제 뇌졸중인 사람은 {n_test_pos:,}명입니다.")

left, right = st.columns(2)
for box, name in zip((left, right), (LOGI_NAME, TREE_NAME)):
    with box:
        st.markdown(f"**{name}**")
        st.plotly_chart(confusion_figure(cells[name]), use_container_width=True)
st.caption(
    "행은 실제, 열은 예측입니다. TP는 뇌졸중을 맞힌 사람, FN은 놓친 환자, "
    "FP는 헛짚은 사람, TN은 아님을 맞힌 사람입니다. 파란 칸은 맞힌 칸, 빨간 칸은 틀린 칸입니다."
)

# ---------- 지표 표 ----------
st.subheader("2. 네 가지 지표")
rows = []
for name in (LOGI_NAME, TREE_NAME, BASE_NAME):
    m = metrics_from(cells[name])
    rows.append(
        {
            "모델": name,
            "정확도": pct(m["정확도"]),
            "재현율": pct(m["재현율"]),
            "정밀도": pct(m["정밀도"]),
            "F1": pct(m["F1"]),
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "정확도 = (TP+TN) ÷ 전체 사람 수, 재현율 = TP ÷ 실제 뇌졸중인 사람 수(TP+FN), "
    "정밀도 = TP ÷ 뇌졸중이라 예측한 사람 수(TP+FP), "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    f"뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 '{NA_TEXT}'입니다."
)

# ---------- 틀린 사람 목록 ----------
st.subheader("3. 틀린 사람 목록")
chosen = st.selectbox("어느 모델의 목록을 볼까요?", [LOGI_NAME, TREE_NAME])
pred = predictions[chosen]
missed = test_df[(y_test == 1) & (pred == 0)]  # 놓친 환자(FN)
wrong_guess = test_df[(y_test == 0) & (pred == 1)]  # 헛짚은 사람(FP)

list_left, list_right = st.columns(2)
with list_left:
    show_list("놓친 환자 (실제 뇌졸중인데 아니라고 예측)", missed)
with list_right:
    show_list("헛짚은 사람 (아닌데 뇌졸중이라고 예측)", wrong_guess)

c = cells[chosen]
st.write(
    f"{chosen}의 두 목록 사람 수는 혼동행렬의 FN 칸({c['FN']:,}명)과 "
    f"FP 칸({c['FP']:,}명)과 같습니다: 놓친 환자 {len(missed):,}명, 헛짚은 사람 {len(wrong_guess):,}명."
)
