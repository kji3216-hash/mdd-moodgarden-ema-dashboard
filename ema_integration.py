"""
MoodGarden EMA integration helpers for the MDD triangle Streamlit app.

The functions in this module treat MoodGarden values as self-report anchors,
not direct neural measurements. They convert repeated daily ratings into
research/education-oriented slider estimates for the MDD circuit model.
"""

from __future__ import annotations

import json
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


RESPONSE_KEYS = ["mood", "anhedonia", "energy", "hopelessness", "rumination"]
CONTEXT_KEYS = [
    "sleepPoor",
    "socialActivity",
    "exercise",
    "stressEvent",
    "medicationTaken",
    "ateMeal",
    "alcohol",
    "therapy",
]
WINDOW_ORDER = ["morning", "afternoon", "evening"]
WINDOW_LABELS = {"morning": "오전", "afternoon": "오후", "evening": "저녁"}


def empty_moodgarden_frame() -> pd.DataFrame:
    columns = [
        "id",
        "schemaVersion",
        "localDate",
        "day",
        "window",
        "scheduledAt",
        "submittedAt",
        "timestamp",
        "timezoneOffset",
        "outOfWindow",
        *RESPONSE_KEYS,
        *CONTEXT_KEYS,
        "note",
        "_window_rank",
        "_submitted_sort",
    ]
    return pd.DataFrame(columns=columns)


@dataclass
class EMASummary:
    total_checkins: int
    unique_days: int
    average_per_day: float
    completion_rate: float
    recent_days: int
    risk_score: float
    mood_autocorrelation: Optional[float]
    mood_variability: float
    trend_label: str
    warning_label: str


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _clip100(value: float) -> int:
    return int(round(float(np.clip(value, 0, 100))))


def parse_moodgarden_csv(uploaded_file) -> pd.DataFrame:
    """Parse a MoodGarden CSV export into a normalized DataFrame."""
    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required = {"localDate", "window", *RESPONSE_KEYS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"MoodGarden CSV 필수 컬럼이 없습니다: {', '.join(missing)}")

    if "submittedAt" not in df.columns:
        df["submittedAt"] = ""
    if "outOfWindow" not in df.columns:
        df["outOfWindow"] = False
    if "note" not in df.columns:
        df["note"] = ""

    df["localDate"] = pd.to_datetime(df["localDate"], errors="coerce")
    df = df.dropna(subset=["localDate"]).copy()
    df["day"] = df["localDate"].dt.strftime("%Y-%m-%d")
    df["window"] = df["window"].astype(str).str.strip()

    for key in RESPONSE_KEYS:
        df[key] = pd.to_numeric(df[key], errors="coerce").clip(0, 100)

    for key in CONTEXT_KEYS:
        if key not in df.columns:
            df[key] = False
        df[key] = df[key].map(_to_bool)

    df["outOfWindow"] = df["outOfWindow"].map(_to_bool)
    df = df.dropna(subset=RESPONSE_KEYS)

    window_rank = {name: i for i, name in enumerate(WINDOW_ORDER)}
    df["_window_rank"] = df["window"].map(window_rank).fillna(99)
    df["_submitted_sort"] = df["submittedAt"].fillna("").astype(str)
    df = df.sort_values(["localDate", "_window_rank", "_submitted_sort"]).reset_index(drop=True)
    return df


def parse_moodgarden_json_file(path: str) -> pd.DataFrame:
    """Parse the live MoodGarden JSON bridge file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    checkins = payload.get("checkins", []) if isinstance(payload, dict) else payload
    if not isinstance(checkins, list):
        raise ValueError("MoodGarden JSON에 checkins 배열이 없습니다.")

    rows = []
    for item in checkins:
        responses = item.get("responses", {}) or {}
        context = item.get("context", {}) or {}
        row = {
            "id": item.get("id", ""),
            "schemaVersion": item.get("schemaVersion", payload.get("schemaVersion", 2) if isinstance(payload, dict) else 2),
            "localDate": item.get("localDate", ""),
            "window": item.get("window", ""),
            "scheduledAt": item.get("scheduledAt", ""),
            "submittedAt": item.get("submittedAt") or item.get("timestamp", ""),
            "timestamp": item.get("timestamp") or item.get("submittedAt", ""),
            "timezoneOffset": item.get("timezoneOffset", ""),
            "outOfWindow": item.get("outOfWindow", False),
            "note": item.get("note", ""),
        }
        for key in RESPONSE_KEYS:
            row[key] = responses.get(key, np.nan)
        for key in CONTEXT_KEYS:
            row[key] = context.get(key, False)
        rows.append(row)

    if not rows:
        return empty_moodgarden_frame()

    return _normalize_moodgarden_frame(pd.DataFrame(rows))


def latest_moodgarden_csv(directory: str) -> Optional[str]:
    candidates = [
        path for path in glob.glob(os.path.join(directory, "moodgarden_*.csv"))
        if not path.endswith("_live.csv")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _normalize_moodgarden_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize either CSV or JSON-shaped MoodGarden rows."""
    df.columns = [str(c).strip() for c in df.columns]

    required = {"localDate", "window", *RESPONSE_KEYS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"MoodGarden 필수 컬럼이 없습니다: {', '.join(missing)}")

    if "submittedAt" not in df.columns:
        df["submittedAt"] = ""
    if "outOfWindow" not in df.columns:
        df["outOfWindow"] = False
    if "note" not in df.columns:
        df["note"] = ""

    df["localDate"] = pd.to_datetime(df["localDate"], errors="coerce")
    df = df.dropna(subset=["localDate"]).copy()
    df["day"] = df["localDate"].dt.strftime("%Y-%m-%d")
    df["window"] = df["window"].astype(str).str.strip()

    for key in RESPONSE_KEYS:
        df[key] = pd.to_numeric(df[key], errors="coerce").clip(0, 100)

    for key in CONTEXT_KEYS:
        if key not in df.columns:
            df[key] = False
        df[key] = df[key].map(_to_bool)

    df["outOfWindow"] = df["outOfWindow"].map(_to_bool)
    df = df.dropna(subset=RESPONSE_KEYS)

    window_rank = {name: i for i, name in enumerate(WINDOW_ORDER)}
    df["_window_rank"] = df["window"].map(window_rank).fillna(99)
    df["_submitted_sort"] = df["submittedAt"].fillna("").astype(str)
    df = df.sort_values(["localDate", "_window_rank", "_submitted_sort"]).reset_index(drop=True)
    return df


def add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["depressed_mood"] = 100 - out["mood"]
    out["anhedonia_risk"] = 100 - out["anhedonia"]
    out["fatigue_risk"] = 100 - out["energy"]
    out["hopelessness_risk"] = 100 - out["hopelessness"]
    out["rumination_risk"] = out["rumination"]
    out["mdd_risk_index"] = out[
        ["depressed_mood", "anhedonia_risk", "fatigue_risk", "hopelessness_risk", "rumination_risk"]
    ].mean(axis=1)
    return out


def recent_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if df.empty:
        return df
    max_day = df["localDate"].max()
    min_day = max_day - pd.Timedelta(days=max(days - 1, 0))
    return df[df["localDate"] >= min_day].copy()


def ema_to_slider_values(df: pd.DataFrame, days: int = 7) -> Dict[str, int]:
    """Map recent MoodGarden EMA values to MDD Streamlit slider defaults."""
    risk = add_risk_columns(recent_window(df, days))
    if risk.empty:
        return {}

    means = risk.mean(numeric_only=True)
    sleep_rate = float(risk["sleepPoor"].mean()) if "sleepPoor" in risk else 0
    stress_rate = float(risk["stressEvent"].mean()) if "stressEvent" in risk else 0
    exercise_rate = float(risk["exercise"].mean()) if "exercise" in risk else 0
    med_rate = float(risk["medicationTaken"].mean()) if "medicationTaken" in risk else 0
    therapy_rate = float(risk["therapy"].mean()) if "therapy" in risk else 0
    social_rate = float(risk["socialActivity"].mean()) if "socialActivity" in risk else 0

    depressed = means["depressed_mood"]
    anhedonia = means["anhedonia_risk"]
    fatigue = means["fatigue_risk"]
    hopelessness = means["hopelessness_risk"]
    rumination = means["rumination_risk"]

    return {
        "sl_dmn": _clip100(0.82 * rumination + 0.18 * depressed),
        "sl_reward": _clip100(0.82 * anhedonia + 18 * (1 - exercise_rate) - 8 * social_rate),
        "sl_sgacc": _clip100(0.68 * hopelessness + 0.22 * depressed + 0.10 * rumination),
        "sl_amygdala": _clip100(0.58 * depressed + 28 * stress_rate + 8 * rumination / 100),
        "sl_sn": _clip100(0.45 * rumination + 0.30 * stress_rate * 100 + 0.25 * fatigue),
        "sl_pfc": _clip100(100 - (0.62 * fatigue + 0.25 * rumination + 0.13 * depressed)),
        "sl_cortisol": _clip100(30 + 38 * sleep_rate + 30 * stress_rate + 8 * fatigue / 100),
        "sl_inflammation": _clip100(25 + 12 * stress_rate + 8 * sleep_rate + 8 * (1 - exercise_rate)),
        "sl_antidepressant": _clip100(70 * med_rate + 30 * therapy_rate),
    }


def daily_averages(df: pd.DataFrame) -> pd.DataFrame:
    risk = add_risk_columns(df)
    grouped = risk.groupby("day", as_index=False).agg(
        mood=("mood", "mean"),
        anhedonia=("anhedonia", "mean"),
        energy=("energy", "mean"),
        hopelessness=("hopelessness", "mean"),
        rumination=("rumination", "mean"),
        mdd_risk_index=("mdd_risk_index", "mean"),
        count=("mood", "count"),
    )
    return grouped.sort_values("day")


def window_averages(df: pd.DataFrame) -> pd.DataFrame:
    risk = add_risk_columns(df)
    grouped = risk.groupby("window", as_index=False).agg(
        mood=("mood", "mean"),
        mdd_risk_index=("mdd_risk_index", "mean"),
        rumination=("rumination", "mean"),
        count=("mood", "count"),
    )
    grouped["_rank"] = grouped["window"].map({name: i for i, name in enumerate(WINDOW_ORDER)}).fillna(99)
    return grouped.sort_values("_rank").drop(columns=["_rank"])


def mood_autocorrelation(df: pd.DataFrame) -> Optional[float]:
    if len(df) < 4:
        return None
    values = df["mood"].astype(float).to_numpy()
    if np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return None
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def trend_label(daily: pd.DataFrame) -> str:
    if len(daily) < 3:
        return "데이터 부족"
    x = np.arange(len(daily))
    slope = float(np.polyfit(x, daily["mdd_risk_index"].to_numpy(), 1)[0])
    if slope > 2:
        return "위험 상승"
    if slope < -2:
        return "회복 방향"
    return "대체로 안정"


def summarize_ema(df: pd.DataFrame, recent_days: int = 7) -> EMASummary:
    risk = add_risk_columns(df)
    recent = add_risk_columns(recent_window(df, recent_days))
    daily = daily_averages(recent)
    unique_days = int(df["day"].nunique())
    expected = unique_days * len(WINDOW_ORDER)
    completion = min(1.0, len(df) / expected) if expected else 0.0
    risk_score = float(recent["mdd_risk_index"].mean()) if not recent.empty else 0.0
    variability = float(daily["mdd_risk_index"].std(ddof=0)) if len(daily) >= 2 else 0.0
    autocorr = mood_autocorrelation(recent)

    warnings: List[str] = []
    if risk_score >= 65:
        warnings.append("최근 위험 지수 높음")
    if variability >= 12:
        warnings.append("일별 변동성 큼")
    if autocorr is not None and autocorr >= 0.45:
        warnings.append("정서 관성 높음")

    return EMASummary(
        total_checkins=int(len(df)),
        unique_days=unique_days,
        average_per_day=float(len(df) / unique_days) if unique_days else 0.0,
        completion_rate=float(completion),
        recent_days=recent_days,
        risk_score=risk_score,
        mood_autocorrelation=autocorr,
        mood_variability=variability,
        trend_label=trend_label(daily),
        warning_label=", ".join(warnings) if warnings else "뚜렷한 조기 경고 없음",
    )


def context_summary(df: pd.DataFrame) -> pd.DataFrame:
    risk = add_risk_columns(df)
    rows = []
    labels = {
        "sleepPoor": "수면 질 저하",
        "socialActivity": "사회활동",
        "exercise": "운동",
        "stressEvent": "스트레스 사건",
        "medicationTaken": "약물 복용",
        "ateMeal": "식사",
        "alcohol": "음주",
        "therapy": "치료/상담",
    }
    for key in CONTEXT_KEYS:
        yes = risk[risk[key]]
        no = risk[~risk[key]]
        if yes.empty:
            continue
        rows.append(
            {
                "context": labels.get(key, key),
                "n": int(len(yes)),
                "risk_when_yes": float(yes["mdd_risk_index"].mean()),
                "risk_when_no": float(no["mdd_risk_index"].mean()) if not no.empty else np.nan,
                "delta": float(yes["mdd_risk_index"].mean() - no["mdd_risk_index"].mean())
                if not no.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False, na_position="last") if rows else pd.DataFrame()


def lifestyle_insights(df: pd.DataFrame, limit: int = 4) -> List[Dict[str, object]]:
    ctx = context_summary(df)
    if ctx.empty:
        return []

    items = []
    for _, row in ctx.dropna(subset=["delta"]).iterrows():
        delta = float(row["delta"])
        direction = "위험 증가" if delta > 0 else "보호 관련"
        circuit = "HPA/스트레스"
        if "수면" in row["context"]:
            circuit = "HPA 축 · 수면/코르티솔"
        elif "스트레스" in row["context"]:
            circuit = "편도체 · Salience Network"
        elif "운동" in row["context"]:
            circuit = "보상 회로 · BDNF"
        elif "사회" in row["context"]:
            circuit = "보상/사회 회로"
        elif "치료" in row["context"] or "약물" in row["context"]:
            circuit = "치료 반응 조절"
        elif "음주" in row["context"]:
            circuit = "수면/HPA 교란"

        items.append(
            {
                "context": row["context"],
                "n": int(row["n"]),
                "delta": delta,
                "direction": direction,
                "circuit": circuit,
                "summary": f"{row['context']}이 표시된 체크인에서 MDD 위험 지수가 {abs(delta):.1f}점 {'높았습니다' if delta > 0 else '낮았습니다'}.",
            }
        )

    return sorted(items, key=lambda item: abs(float(item["delta"])), reverse=True)[:limit]


def phase_points(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    daily = daily_averages(recent_window(df, days))
    if daily.empty:
        return daily
    points = daily.copy()
    points["factor_x"] = (points["mdd_risk_index"] / 100).clip(0, 1)
    points["label"] = points["day"] + "<br>위험 지수 " + points["mdd_risk_index"].round(1).astype(str)
    return points


def add_ema_phase_overlay(fig: go.Figure, df: pd.DataFrame, circuit_name: str, days: int = 14) -> go.Figure:
    points = phase_points(df, days)
    if points.empty:
        return fig
    y = points["factor_x"]
    if circuit_name == "dmn_overactivity":
        y = (points["rumination"] / 100).clip(0, 1)
    elif circuit_name == "reward_circuit_dysfunction":
        y = ((100 - points["anhedonia"]) / 100).clip(0, 1)
    elif circuit_name == "sgacc_hyperactivity":
        y = ((100 - points["hopelessness"]) / 100).clip(0, 1)

    fig.add_trace(
        go.Scatter(
            x=points["factor_x"],
            y=y,
            mode="lines+markers",
            name="🌱 MoodGarden EMA 궤적",
            marker=dict(size=9, color="#7BED9F", line=dict(color="white", width=1)),
            line=dict(color="#7BED9F", width=2),
            text=points["label"],
            hovertemplate="%{text}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
        )
    )
    return fig


def plot_ema_trajectory(df: pd.DataFrame) -> go.Figure:
    daily = daily_averages(df)
    fig = go.Figure()
    for key, label, color in [
        ("mood", "기분", "#2ED573"),
        ("anhedonia", "즐거움/흥미", "#FFD166"),
        ("energy", "에너지", "#70A1FF"),
        ("hopelessness", "희망감", "#A29BFE"),
        ("rumination", "반추", "#FF6B81"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=daily["day"],
                y=daily[key],
                mode="lines+markers",
                name=label,
                line=dict(width=2.5, color=color),
            )
        )
    fig.update_layout(
        height=420,
        yaxis=dict(range=[0, 100], title="VAS 점수"),
        xaxis_title="날짜",
        hovermode="x unified",
        paper_bgcolor="rgba(15,15,35,1)",
        plot_bgcolor="rgba(20,20,40,1)",
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def plot_risk_timeline(df: pd.DataFrame) -> go.Figure:
    daily = daily_averages(df)
    fig = go.Figure()
    colors = [
        "#2ED573" if v < 45 else "#FFA502" if v < 65 else "#FF4757"
        for v in daily["mdd_risk_index"]
    ]
    fig.add_trace(
        go.Bar(
            x=daily["day"],
            y=daily["mdd_risk_index"],
            name="MDD 위험 지수",
            marker_color=colors,
            hovertemplate="%{x}<br>위험 지수 %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["day"],
            y=daily["mood"],
            mode="lines+markers",
            name="기분",
            line=dict(color="#70A1FF", width=2.5),
            marker=dict(size=7),
            hovertemplate="%{x}<br>기분 %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hrect(y0=0, y1=45, fillcolor="rgba(46,213,115,0.06)", line_width=0)
    fig.add_hrect(y0=45, y1=65, fillcolor="rgba(255,165,2,0.06)", line_width=0)
    fig.add_hrect(y0=65, y1=100, fillcolor="rgba(255,71,87,0.06)", line_width=0)
    fig.update_layout(
        height=360,
        yaxis=dict(range=[0, 100], title="점수", gridcolor="rgba(120,120,140,0.18)"),
        xaxis=dict(title="", tickangle=0),
        hovermode="x unified",
        paper_bgcolor="rgba(15,15,35,1)",
        plot_bgcolor="rgba(20,20,40,1)",
        font=dict(color="white", size=12),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig


def plot_context_impact(df: pd.DataFrame) -> go.Figure:
    ctx = context_summary(df)
    fig = go.Figure()
    if ctx.empty:
        return fig

    ctx = ctx.dropna(subset=["delta"]).copy()
    ctx["abs_delta"] = ctx["delta"].abs()
    ctx = ctx.sort_values("abs_delta", ascending=True).tail(8)
    colors = ["#FF6B81" if v > 0 else "#2ED573" for v in ctx["delta"]]
    fig.add_trace(
        go.Bar(
            x=ctx["delta"],
            y=ctx["context"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}" for v in ctx["delta"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>위험 지수 차이 %{x:+.1f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.45)", width=1))
    fig.update_layout(
        height=340,
        xaxis=dict(title="해당 시 위험 지수 변화", gridcolor="rgba(120,120,140,0.18)"),
        yaxis=dict(title="", automargin=True),
        paper_bgcolor="rgba(15,15,35,1)",
        plot_bgcolor="rgba(20,20,40,1)",
        font=dict(color="white", size=12),
        margin=dict(l=10, r=50, t=25, b=10),
        showlegend=False,
    )
    return fig


def plot_slider_estimates(slider_values: Dict[str, int]) -> go.Figure:
    label_map = {
        "sl_dmn": "DMN 반추",
        "sl_reward": "보상 저기능",
        "sl_sgacc": "sgACC 고착",
        "sl_amygdala": "편도체 감도",
        "sl_sn": "SN 전환",
        "sl_pfc": "PFC 조절",
        "sl_cortisol": "HPA/코르티솔",
        "sl_inflammation": "염증",
        "sl_antidepressant": "치료 반응",
    }
    items = [(label_map.get(k, k), v, k) for k, v in slider_values.items()]
    items.sort(key=lambda item: item[1])
    colors = ["#70A1FF" if key == "sl_pfc" else "#2ED573" if v < 45 else "#FFA502" if v < 65 else "#FF6B81" for _, v, key in items]
    fig = go.Figure(
        go.Bar(
            x=[v for _, v, _ in items],
            y=[label for label, _, _ in items],
            orientation="h",
            marker_color=colors,
            text=[str(v) for _, v, _ in items],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=340,
        xaxis=dict(range=[0, 105], title="EMA 기반 추정치", gridcolor="rgba(120,120,140,0.18)"),
        yaxis=dict(title="", automargin=True),
        paper_bgcolor="rgba(15,15,35,1)",
        plot_bgcolor="rgba(20,20,40,1)",
        font=dict(color="white", size=12),
        margin=dict(l=10, r=45, t=25, b=10),
        showlegend=False,
    )
    return fig


def plot_window_patterns(df: pd.DataFrame) -> go.Figure:
    win = window_averages(df)
    labels = [WINDOW_LABELS.get(w, w) for w in win["window"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=win["mood"], name="기분", marker_color="#2ED573"))
    fig.add_trace(go.Bar(x=labels, y=win["mdd_risk_index"], name="위험 지수", marker_color="#FF6B81"))
    fig.update_layout(
        height=360,
        yaxis=dict(range=[0, 100], title="평균 점수"),
        barmode="group",
        paper_bgcolor="rgba(15,15,35,1)",
        plot_bgcolor="rgba(20,20,40,1)",
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.1),
    )
    return fig
