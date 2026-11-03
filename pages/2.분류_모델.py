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
PAGE_TITLE = "분류 모델"
PAGE_ICON = "🤖"
DATA_URL = "https://raw.githubusercontent.com/lim7887/data/main/stroke.csv"

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}", page_icon=PAGE_ICON, layout="wide"
)

# 열 이름 -> 우리말 이름 (화면에는 우리말 이름만 보여 준다)
FEATURES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}  # 0/1 속성
DEFAULT_NAMES = ["나이", "평균 혈당", "고혈압", "심장병"]  # 처음에는 bmi를 뺀 넷
SUBJECT = {  # 의사결정트리 질문에 쓰는 '~이/가'
    "age": "나이가",
    "avg_glucose_level": "평균 혈당이",
    "bmi": "체질량지수가",
}
BINARY_QUESTION = {
    "hypertension": "고혈압이 없나요?",
    "heart_disease": "심장병이 없나요?",
}

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "다수 클래스 기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

NEG_LABEL, POS_LABEL = "뇌졸중 아님", "뇌졸중"
POINT_COLORS = {0: "#4C78A8", 1: "#E45756"}  # 점 색
REGION_COLORS = {0: "#DCE8F5", 1: "#F8DCDC"}  # 옅은 칸 색

st.markdown(
    """
    <style>
    .acc-card {
        border: 1px solid rgba(128,128,128,0.35);
        background: rgba(128,128,128,0.08);
        border-radius: 12px;
        padding: 16px 18px;
        height: 100%;
    }
    .acc-title { font-size: 0.95rem; font-weight: 600; min-height: 3.2em; }
    .acc-big { font-size: 3rem; font-weight: 700; line-height: 1.1; }
    .acc-cap { font-size: 0.85rem; opacity: 0.7; margin-bottom: 10px; }
    .acc-small { display: flex; gap: 18px; font-size: 0.85rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def acc_card(title: str, train_acc: float, test_acc: float) -> str:
    return (
        '<div class="acc-card">'
        f'<div class="acc-title">{title}</div>'
        f'<div class="acc-big">{test_acc * 100:.2f}%</div>'
        '<div class="acc-cap">테스트 데이터 정확도</div>'
        '<div class="acc-small">'
        f"<span>훈련 데이터 {train_acc * 100:.2f}%</span>"
        f"<span>테스트 데이터 {test_acc * 100:.2f}%</span>"
        "</div></div>"
    )


def axis_range(values: pd.Series) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.1 or 0.5
    return lo - pad, hi + pad


def tree_question(col: str, threshold: float) -> str:
    if col in BINARY:
        return BINARY_QUESTION[col]
    return f"{SUBJECT[col]} {threshold:.1f} 이하인가요?"


def build_dot(tree: DecisionTreeClassifier, cols: list, X_train, y_train) -> str:
    """훈련 데이터가 각 마디를 지나간 사람 수까지 적은 DOT 문자열을 만든다."""
    t = tree.tree_
    path = tree.decision_path(X_train)  # (사람 x 마디) 표
    n_node = np.asarray(path.sum(axis=0)).ravel()
    n_pos = np.asarray(path.T.dot(y_train.to_numpy())).ravel()

    font = "Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"
    lines = [
        "digraph Tree {",
        "rankdir=TB;",
        f'node [shape=box, style="rounded,filled", color="#888888", '
        f'fontcolor="#222222", fontname="{font}"];',
        f'edge [fontname="{font}"];',
    ]
    for i in range(t.node_count):
        n, pos = int(n_node[i]), int(n_pos[i])
        info = f"훈련 {n:,}명\\n뇌졸중 {pos:,}명 ({pos / n * 100:.1f}%)"
        if t.children_left[i] == -1:  # 답을 내는 마디
            answer = int(t.value[i][0].argmax())
            word = POS_LABEL if answer == 1 else "아님"
            label = f"답: {word}\\n{info}"
            fill = REGION_COLORS[answer]
        else:
            col = cols[t.feature[i]]
            label = f"{tree_question(col, t.threshold[i])}\\n{info}"
            fill = "#F5F5F5"
        lines.append(f'n{i} [label="{label}", fillcolor="{fill}"];')
    for i in range(t.node_count):
        if t.children_left[i] != -1:
            lines.append(f'n{i} -> n{t.children_left[i]} [label="예"];')
            lines.append(f'n{i} -> n{t.children_right[i]} [label="아니요"];')
    lines.append("}")
    return "\n".join(lines)


st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · 입력 속성으로 뇌졸중을 맞히는 두 모델을 비교합니다")
st.markdown(
    "`stroke`가 1이면 뇌졸중, 0이면 아님이며, **뇌졸중을 양성**으로 둡니다. "
    f"{LOGI_NAME}과 {TREE_NAME}을 만들어 정확도를 견줘 봅니다."
)

# ---------- 입력 속성 고르기 ----------
chosen_names = st.multiselect(
    "입력으로 사용할 속성 (두 개 이상)",
    list(FEATURES.values()),
    default=DEFAULT_NAMES,
)
cols = [c for c, name in FEATURES.items() if name in chosen_names]
if len(cols) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개 이상 고르면 모델이 만들어집니다.")
    st.stop()
names = [FEATURES[c] for c in cols]

try:
    raw = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ---------- 훈련용/테스트용 나누기 ----------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3  # 열 명씩 묶어 앞 세 명이 테스트용
X_train = df.loc[~is_test, cols].copy()
X_test = df.loc[is_test, cols].copy()
y_train = df.loc[~is_test, "stroke"]
y_test = df.loc[is_test, "stroke"]

if "bmi" in cols:  # 체질량지수를 고른 경우에만, 훈련용 중앙값으로 채운다
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

st.caption(
    "번호 순으로 정렬한 뒤 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했습니다. "
    f"훈련용 {len(X_train):,}명, 테스트용 {len(X_test):,}명입니다. "
    "크기 맞추기(표준화)는 훈련용으로만 합니다."
    + (
        f" 체질량지수의 빈 값은 훈련용 중앙값({bmi_median:.2f})으로 채웠습니다."
        if "bmi" in cols
        else ""
    )
)

# ---------- 모델 만들기 ----------
logi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logi.fit(X_train, y_train)
# 의사결정트리는 값의 크기 맞추기와 상관없이 같은 질문을 하므로, 질문이 읽히도록 원래 값 그대로 쓴다.
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

# ---------- 정확도 카드 ----------
st.subheader("1. 정확도 비교")
cards = [
    (LOGI_NAME, logi),
    (TREE_NAME, tree),
    (BASE_NAME, base),
]
card_cols = st.columns(3)
for box, (title, model) in zip(card_cols, cards):
    with box:
        st.markdown(
            acc_card(title, model.score(X_train, y_train), model.score(X_test, y_test)),
            unsafe_allow_html=True,
        )
st.caption(
    "큰 숫자는 테스트 데이터의 정확도입니다. 뇌졸중이 드물어서 기준 모델도 정확도가 높게 나오니, "
    "두 모델이 기준 모델보다 나은지 함께 살펴보세요."
)

# ---------- 산점도 ----------
st.subheader("2. 테스트 데이터 산점도와 두 모델의 경계")
sel_x, sel_y = st.columns(2)
with sel_x:
    x_name = st.selectbox("가로축", names, index=0)
with sel_y:
    y_options = [n for n in names if n != x_name]
    y_name = st.selectbox("세로축", y_options, index=0)
x_col = cols[names.index(x_name)]
y_col = cols[names.index(y_name)]

median_test = X_test.median()
x_lo, x_hi = axis_range(X_test[x_col])
y_lo, y_hi = axis_range(X_test[y_col])
N = 200
xs = np.linspace(x_lo, x_hi, N)
ys = np.linspace(y_lo, y_hi, N)
gx, gy = np.meshgrid(xs, ys)
grid = pd.DataFrame({c: np.full(gx.size, median_test[c]) for c in cols})
grid[x_col] = gx.ravel()
grid[y_col] = gy.ravel()
grid = grid[cols]

prob = logi.predict_proba(grid)[:, 1].reshape(gx.shape)  # 뇌졸중일 확률
tree_pred = tree.predict(grid).reshape(gx.shape)
_, leaf_rank = np.unique(tree.apply(grid), return_inverse=True)
leaf_rank = leaf_rank.reshape(gx.shape)
n_cells = int(leaf_rank.max()) + 1

fig = go.Figure()

# (1) 의사결정트리가 나눈 칸: 답에 따라 옅은 색
fig.add_trace(
    go.Heatmap(
        x=xs,
        y=ys,
        z=tree_pred,
        zmin=0,
        zmax=1,
        colorscale=[
            [0, REGION_COLORS[0]],
            [0.5, REGION_COLORS[0]],
            [0.5, REGION_COLORS[1]],
            [1, REGION_COLORS[1]],
        ],
        showscale=False,
        hoverinfo="skip",
        showlegend=False,
    )
)
# 칸과 칸 사이의 경계(옅은 회색 가는 선)
if n_cells > 1:
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=leaf_rank,
            contours=dict(start=0.5, end=n_cells - 1.5, size=1, coloring="lines"),
            line=dict(color="rgba(120,120,120,0.8)", width=1),
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )

# (2) 로지스틱 회귀가 0.5로 가르는 선
line_visible = bool((prob >= 0.5).any() and (prob < 0.5).any())
if line_visible:
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=prob,
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
            line=dict(color="#111111", width=3),
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )

# (3) 테스트 데이터 점 (실제 뇌졸중 여부로 색을 나눔)
rng = np.random.default_rng(0)


def shown(col: str, mask) -> np.ndarray:
    v = X_test.loc[mask, col].to_numpy(dtype=float)
    if col in BINARY:  # 0/1 속성은 점이 겹치지 않게 살짝 흩어서 보여 줌
        v = v + rng.uniform(-0.05, 0.05, size=len(v))
    return v


for value, label in [(0, NEG_LABEL), (1, POS_LABEL)]:
    mask = (y_test == value).to_numpy()
    fig.add_trace(
        go.Scatter(
            x=shown(x_col, mask),
            y=shown(y_col, mask),
            mode="markers",
            name=f"실제 {label}",
            marker=dict(
                color=POINT_COLORS[value],
                size=7 if value == 0 else 9,
                opacity=0.55 if value == 0 else 0.95,
                line=dict(color="white", width=0.5),
            ),
        )
    )

# 범례용 안내 표식
if line_visible:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#111111", width=3),
            name=f"{LOGI_NAME} 0.5 경계선",
        )
    )
for value in np.unique(tree_pred):
    word = "뇌졸중" if value == 1 else "아님"
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=14,
                color=REGION_COLORS[int(value)],
                line=dict(color="#999999", width=1),
            ),
            name=f"{TREE_NAME}이 '{word}'이라고 답하는 칸",
        )
    )

fig.update_xaxes(title=x_name, range=[x_lo, x_hi])
fig.update_yaxes(title=y_name, range=[y_lo, y_hi])
for axis_col, setter in [(x_col, fig.update_xaxes), (y_col, fig.update_yaxes)]:
    if axis_col in BINARY:
        setter(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(
    height=620,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    margin=dict(t=30),
)
st.plotly_chart(fig, use_container_width=True)

others = [c for c in cols if c not in (x_col, y_col)]
if others:
    parts = []
    for c in others:
        v = round(float(median_test[c]), 2)
        if c in BINARY:
            parts.append(f"{FEATURES[c]} = {v:g}({'있음' if v >= 0.5 else '없음'})")
        else:
            parts.append(f"{FEATURES[c]} = {v:g}")
    st.write(
        "그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 두 모델의 답을 계산했습니다: "
        + ", ".join(parts)
        + "."
    )
else:
    st.write("고른 속성이 둘뿐이라, 따로 중앙값에 세워 둔 속성은 없습니다.")

if line_visible:
    st.write(
        f"검은 실선은 {LOGI_NAME}이 뇌졸중일 확률을 0.5로 가르는 선입니다. "
        "선의 한쪽은 '뇌졸중', 다른 쪽은 '아님'이라고 답합니다."
    )
else:
    side = "낮아서 모두 '아님'" if (prob < 0.5).all() else "높아서 모두 '뇌졸중'"
    st.write(
        f"{LOGI_NAME}의 0.5 경계선은 이 그림 범위 밖에 있어 그려지지 않았습니다. "
        f"그림 안의 모든 자리에서 뇌졸중일 확률이 0.5보다 {side}이라고 답합니다."
    )
st.write(
    f"옅은 색 칸은 {TREE_NAME}이 나눈 칸입니다. 옅은 파란색 칸은 '아님', "
    "옅은 빨간색 칸은 '뇌졸중'이라고 답하는 자리이고, 회색 가는 선이 칸의 경계입니다."
)
if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병은 0 또는 1이라, 점이 겹치지 않게 살짝 흩어 찍었습니다.")

# ---------- 의사결정트리 그림 ----------
st.subheader("3. 의사결정트리가 던진 질문")
st.caption(
    "마디마다 그 자리에 온 훈련용 사람 수와 그중 실제 뇌졸중인 사람 수, 그 비율을 적었습니다. "
    "'예'는 질문에 그렇다는 쪽, '아니요'는 아니라는 쪽 가지입니다."
)
st.graphviz_chart(build_dot(tree, cols, X_train, y_train), use_container_width=True)

t = tree.tree_
is_leaf = t.children_left == -1
n_leaves = int(is_leaf.sum())
n_no = int(sum(1 for i in np.where(is_leaf)[0] if t.value[i][0].argmax() == 0))
used = sorted({int(f) for f in t.feature if f >= 0})
st.write(
    f"답을 내는 마디는 모두 {n_leaves}칸이고, 그중 {n_no}칸이 '아님'이라고 답합니다."
)
if used:
    st.write(
        "고른 속성 가운데 이 나무가 실제로 물은 것은 "
        + ", ".join(names[i] for i in used)
        + "입니다."
    )
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것은 없습니다. 처음부터 답을 냈습니다.")
