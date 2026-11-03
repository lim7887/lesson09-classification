import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "성능 높이기"
PAGE_ICON = "⚖️"
DATA_URL = "https://raw.githubusercontent.com/lim7887/data/main/stroke.csv"

st.set_page_config(
    page_title=f"{PAGE_TITLE} · {APP_TITLE}", page_icon=PAGE_ICON, layout="wide"
)

FEATURES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}
DEFAULT_NAMES = ["나이", "평균 혈당", "고혈압", "심장병"]

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BEFORE = "가중치를 맞추기 전"
AFTER = "가중치를 맞춘 뒤"
NA_TEXT = "계산할 수 없음"

NEG_LABEL, POS_LABEL = "뇌졸중 아님", "뇌졸중"
POINT_COLORS = {0: "#4C78A8", 1: "#E45756"}
LINE_COLORS = {BEFORE: "#2CA02C", AFTER: "#FF7F0E"}  # 경계선·평면 색
EPS = 1e-12


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 채점 ----------
def count_cells(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
    }


def metrics_from(c: dict) -> dict:
    total = c["TP"] + c["FN"] + c["FP"] + c["TN"]
    actual_pos = c["TP"] + c["FN"]
    pred_pos = c["TP"] + c["FP"]
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


def compare_table(before: dict, after: dict) -> pd.DataFrame:
    """네 칸과 네 지표를 전·후로 나란히 놓은 표."""
    rows = [
        ("TP (찾아낸 환자)", f"{before['TP']:,}명", f"{after['TP']:,}명"),
        ("FN (놓친 환자)", f"{before['FN']:,}명", f"{after['FN']:,}명"),
        ("FP (헛짚은 사람)", f"{before['FP']:,}명", f"{after['FP']:,}명"),
        ("TN (아님을 맞힘)", f"{before['TN']:,}명", f"{after['TN']:,}명"),
    ]
    mb, ma = metrics_from(before), metrics_from(after)
    for key in ("정확도", "재현율", "정밀도", "F1"):
        rows.append((key, pct(mb[key]), pct(ma[key])))
    return pd.DataFrame(rows, columns=["항목", BEFORE, AFTER])


# ---------- 로지스틱 회귀의 0.5 경계(선·평면) ----------
def logit_parts(pipe, cols: list) -> tuple:
    """크기를 맞추기 전의 원래 값 기준 계수와 절편을 돌려준다. (logit = 절편 + Σ 계수×값)"""
    sc = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    w = lr.coef_[0] / sc.scale_
    b = float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))
    return pd.Series(w, index=cols), b


def axis_equation(pipe, cols: list, axes: list, median: pd.Series) -> tuple:
    """고른 축 밖의 속성을 중앙값에 세웠을 때 (축 계수들, 상수)를 돌려준다."""
    w, b = logit_parts(pipe, cols)
    const = b + sum(float(w[c]) * float(median[c]) for c in cols if c not in axes)
    return [float(w[a]) for a in axes], const


def clip_line(a, b, c, xlo, xhi, ylo, yhi):
    """a·x + b·y + c = 0 인 직선이 그림 상자 안에 지나가는 두 끝점을 돌려준다. 없으면 None."""
    pts = []
    if abs(b) > EPS:
        for x in (xlo, xhi):
            y = -(c + a * x) / b
            if ylo <= y <= yhi:
                pts.append((x, y))
    if abs(a) > EPS:
        for y in (ylo, yhi):
            x = -(c + b * y) / a
            if xlo <= x <= xhi:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    p0 = pts[0]
    far = max(pts, key=lambda p: (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2)
    if far == p0:
        return None
    return p0, far


def plane_surface(coefs, const, ranges, n=50):
    """a·x + b·y + c·z + const = 0 인 평면 가운데 그림 상자 안에 있는 부분. 없으면 None."""
    k = int(np.argmax(np.abs(coefs)))
    if abs(coefs[k]) < EPS:
        return None
    o1, o2 = [i for i in range(3) if i != k]
    g1 = np.linspace(*ranges[o1], n)
    g2 = np.linspace(*ranges[o2], n)
    G1, G2 = np.meshgrid(g1, g2)
    V = -(const + coefs[o1] * G1 + coefs[o2] * G2) / coefs[k]
    V = np.where((V >= ranges[k][0]) & (V <= ranges[k][1]), V, np.nan)
    if not np.isfinite(V).any():
        return None
    out = [None, None, None]
    out[o1], out[o2], out[k] = G1, G2, V
    return out


def side_word(value: float) -> str:
    """상자 가운데에서 로지스틱 값이 음수면 확률이 0.5보다 낮다."""
    return "낮아서 모두 '아님'" if value < 0 else "높아서 모두 '뇌졸중'"


def axis_range(values: pd.Series) -> tuple:
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.1 or 0.5
    return lo - pad, hi + pad


# ---------- 화면 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · 뇌졸중이 드물다는 점을 모델에 알려 주면 결과가 어떻게 달라질까요?")

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

# ---------- '채점표' 페이지와 같은 방법으로 읽고 나누기 ----------
df = raw.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3  # 열 명씩 묶어 앞 세 명이 테스트용
X_train = df.loc[~is_test, cols].copy()
y_train = df.loc[~is_test, "stroke"]
X_test = df.loc[is_test, cols].copy().reset_index(drop=True)
y_test = df.loc[is_test, "stroke"].to_numpy()

if "bmi" in cols:  # 고른 경우에만, 훈련용 중앙값으로 빈 값 채우기
    bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)


# ---------- 모델: 가중치를 맞추기 전과 뒤 ----------
def make_logi(weighted: bool):
    cw = "balanced" if weighted else None
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight=cw)
    )


def make_tree(weighted: bool):
    cw = "balanced" if weighted else None
    return DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight=cw
    )


models = {
    LOGI_NAME: {BEFORE: make_logi(False), AFTER: make_logi(True)},
    TREE_NAME: {BEFORE: make_tree(False), AFTER: make_tree(True)},
}
cells = {}
for model_name, pair in models.items():
    cells[model_name] = {}
    for stage, model in pair.items():
        model.fit(X_train, y_train)
        cells[model_name][stage] = count_cells(y_test, model.predict(X_test))

n_train_pos = int((y_train == 1).sum())
n_train_neg = int((y_train == 0).sum())
ratio = n_train_neg / n_train_pos if n_train_pos else float("nan")
st.caption(
    f"훈련용 {len(X_train):,}명 가운데 뇌졸중은 {n_train_pos:,}명, 아님은 {n_train_neg:,}명입니다. "
    f"가중치를 맞추면 뇌졸중인 사람 한 명을 아님인 사람 한 명보다 약 {ratio:.1f}배 무겁게 세어, "
    "양쪽의 무게가 같아지게 학습합니다. 의사결정트리의 설정(질문 세 번까지, 마지막 마디 다섯 명 이상, 난수 42)은 그대로입니다."
    + (
        f" 체질량지수의 빈 값은 훈련용 중앙값({bmi_median:.2f})으로 채웠습니다."
        if "bmi" in cols
        else ""
    )
)

# ---------- 1. 전·후 표 ----------
st.subheader("1. 가중치를 맞추기 전과 뒤")
st.write(
    f"테스트용 {len(X_test):,}명 가운데 실제 뇌졸중인 사람은 {int((y_test == 1).sum()):,}명입니다. "
    "뇌졸중을 양성으로 둡니다."
)
left, right = st.columns(2)
for box, model_name in zip((left, right), (LOGI_NAME, TREE_NAME)):
    before, after = cells[model_name][BEFORE], cells[model_name][AFTER]
    with box:
        st.markdown(f"**{model_name}**")
        st.dataframe(compare_table(before, after), hide_index=True, use_container_width=True)
        st.write(
            f"찾아낸 환자 {before['TP']:,}명 → {after['TP']:,}명, "
            f"놓친 환자 {before['FN']:,}명 → {after['FN']:,}명, "
            f"헛짚은 사람 {before['FP']:,}명 → {after['FP']:,}명으로 바뀌었습니다."
        )

# ---------- 2. 2차원 산점도 ----------
logi_before = models[LOGI_NAME][BEFORE]
logi_after = models[LOGI_NAME][AFTER]
median_test = X_test.median()
rng = np.random.default_rng(0)


def shown(col: str, mask) -> np.ndarray:
    v = X_test.loc[mask, col].to_numpy(dtype=float)
    if col in BINARY:  # 0/1 속성은 점이 겹치지 않게 살짝 흩어서 보여 줌
        v = v + rng.uniform(-0.05, 0.05, size=len(v))
    return v


def fixed_values_text(axes: list) -> str:
    others = [c for c in cols if c not in axes]
    if not others:
        return "고른 속성이 축으로 모두 쓰여, 따로 중앙값에 세워 둔 속성은 없습니다."
    parts = []
    for c in others:
        v = round(float(median_test[c]), 2)
        if c in BINARY:
            parts.append(f"{FEATURES[c]} = {v:g}({'있음' if v >= 0.5 else '없음'})")
        else:
            parts.append(f"{FEATURES[c]} = {v:g}")
    return (
        "축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 경계를 계산했습니다: "
        + ", ".join(parts)
        + "."
    )


st.subheader("2. 산점도와 0.5 경계선")
sel_x, sel_y = st.columns(2)
with sel_x:
    x_name = st.selectbox("가로축", names, index=0)
with sel_y:
    y_name = st.selectbox("세로축", [n for n in names if n != x_name], index=0)
x_col = cols[names.index(x_name)]
y_col = cols[names.index(y_name)]
x_lo, x_hi = axis_range(X_test[x_col])
y_lo, y_hi = axis_range(X_test[y_col])

fig = go.Figure()
for value, label in [(0, NEG_LABEL), (1, POS_LABEL)]:
    mask = y_test == value
    fig.add_trace(
        go.Scatter(
            x=shown(x_col, mask),
            y=shown(y_col, mask),
            mode="markers",
            name=f"실제 {label}",
            marker=dict(
                color=POINT_COLORS[value],
                size=7 if value == 0 else 9,
                opacity=0.5 if value == 0 else 0.95,
                line=dict(color="white", width=0.5),
            ),
        )
    )

notes_2d = []
for stage, pipe in [(BEFORE, logi_before), (AFTER, logi_after)]:
    (a, b), c = axis_equation(pipe, cols, [x_col, y_col], median_test)
    seg = clip_line(a, b, c, x_lo, x_hi, y_lo, y_hi)
    if seg is None:
        center = a * (x_lo + x_hi) / 2 + b * (y_lo + y_hi) / 2 + c
        notes_2d.append(
            f"{stage}의 0.5 경계선은 이 그림 범위 밖에 있어 그려지지 않았습니다. "
            f"그림 안의 모든 자리에서 뇌졸중일 확률이 0.5보다 {side_word(center)}이라고 답합니다."
        )
    else:
        (x0, y0), (x1, y1) = seg
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                name=f"{stage} 0.5 경계선",
                line=dict(
                    color=LINE_COLORS[stage],
                    width=4,
                    dash="dash" if stage == BEFORE else "solid",
                ),
            )
        )

fig.update_xaxes(title=x_name, range=[x_lo, x_hi])
fig.update_yaxes(title=y_name, range=[y_lo, y_hi])
for axis_col, setter in [(x_col, fig.update_xaxes), (y_col, fig.update_yaxes)]:
    if axis_col in BINARY:
        setter(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(
    height=600,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    margin=dict(t=30),
)
st.plotly_chart(fig, use_container_width=True)
st.write(
    f"점은 테스트 데이터이고 색은 실제 뇌졸중 여부입니다. 두 선은 {LOGI_NAME}이 "
    f"뇌졸중일 확률을 0.5로 가르는 선으로, {BEFORE}의 선은 초록 점선, {AFTER}의 선은 주황 실선입니다."
)
st.write(fixed_values_text([x_col, y_col]))
for note in notes_2d:
    st.write(note)
if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병은 0 또는 1이라, 점이 겹치지 않게 살짝 흩어 찍었습니다.")

# ---------- 3. 3차원 산점도 ----------
st.subheader("3. 3차원 산점도와 0.5 평면")
if len(cols) < 3:
    st.info("속성을 셋 이상 고르면 3차원 산점도와 0.5 평면이 나타납니다.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3_name = st.selectbox("3차원 가로축(x)", names, index=0)
    with c2:
        y3_name = st.selectbox(
            "3차원 세로축(y)", [n for n in names if n != x3_name], index=0
        )
    with c3:
        z3_name = st.selectbox(
            "3차원 높이축(z)",
            [n for n in names if n not in (x3_name, y3_name)],
            index=0,
        )
    axes3 = [cols[names.index(n)] for n in (x3_name, y3_name, z3_name)]
    ranges3 = [axis_range(X_test[c]) for c in axes3]

    fig3 = go.Figure()
    for value, label in [(0, NEG_LABEL), (1, POS_LABEL)]:
        mask = y_test == value
        fig3.add_trace(
            go.Scatter3d(
                x=shown(axes3[0], mask),
                y=shown(axes3[1], mask),
                z=shown(axes3[2], mask),
                mode="markers",
                name=f"실제 {label}",
                marker=dict(
                    color=POINT_COLORS[value],
                    size=3 if value == 0 else 5,
                    opacity=0.5 if value == 0 else 0.95,
                ),
            )
        )

    notes_3d = []
    for stage, pipe in [(BEFORE, logi_before), (AFTER, logi_after)]:
        coefs, const = axis_equation(pipe, cols, axes3, median_test)
        surf = plane_surface(coefs, const, ranges3)
        if surf is None:
            center = const + sum(
                cf * (lo + hi) / 2 for cf, (lo, hi) in zip(coefs, ranges3)
            )
            notes_3d.append(
                f"{stage}의 0.5 평면은 이 그림 범위 밖에 있어 그려지지 않았습니다. "
                f"그림 안의 모든 자리에서 뇌졸중일 확률이 0.5보다 {side_word(center)}이라고 답합니다."
            )
            continue
        fig3.add_trace(
            go.Surface(
                x=surf[0],
                y=surf[1],
                z=surf[2],
                colorscale=[[0, LINE_COLORS[stage]], [1, LINE_COLORS[stage]]],
                showscale=False,
                opacity=0.45,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # 범례에 평면 색을 보여 주는 표식
        fig3.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="lines",
                line=dict(color=LINE_COLORS[stage], width=8),
                name=f"{stage} 0.5 평면",
            )
        )

    scene = dict(aspectmode="cube")
    for key, axis_name, axis_col, rng_ in zip(
        ("xaxis", "yaxis", "zaxis"),
        (x3_name, y3_name, z3_name),
        axes3,
        ranges3,
    ):
        axis = dict(title=axis_name, range=list(rng_))
        if axis_col in BINARY:
            axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
        scene[key] = axis
    fig3.update_layout(
        height=700,
        scene=scene,
        legend=dict(orientation="h", yanchor="top", y=0.0, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.write(
        f"마우스로 끌면 그림이 돌아가고, 휠로 크기를 바꿀 수 있습니다. 초록 평면은 {BEFORE}, "
        f"주황 평면은 {AFTER}의 {LOGI_NAME} 0.5 평면입니다."
    )
    st.write(fixed_values_text(axes3))
    for note in notes_3d:
        st.write(note)
    if any(c in BINARY for c in axes3):
        st.caption("고혈압·심장병은 0 또는 1이라, 점이 겹치지 않게 살짝 흩어 찍었습니다.")
