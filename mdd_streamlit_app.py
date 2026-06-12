#!/usr/bin/env python3
"""
주요 우울장애(MDD) 유전-뇌-행동 삼각구조 시각화
BMB 프로젝트 - 서울대학교 뇌-마음-행동 교과목

이론적 기반:
  - Yuan et al. (2026) Neuron — Pathological Attractor Model
  - Menon (2011) — Triple Network Model
  - PGC MDD (2025) Cell — 697 loci GWAS
"""

import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np
import os

from ema_integration import (
    add_ema_phase_overlay,
    context_summary,
    ema_to_slider_values,
    latest_moodgarden_csv,
    lifestyle_insights,
    parse_moodgarden_csv,
    parse_moodgarden_json_file,
    plot_context_impact,
    plot_ema_trajectory,
    plot_risk_timeline,
    plot_slider_estimates,
    plot_window_patterns,
    summarize_ema,
)

# ==================== 설정 ====================

st.set_page_config(
    page_title="MDD 유전-뇌-행동 삼각구조",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

WARNING_TEXT = """
⚠️ **중요 면책 조항**
- 이 도구는 **진단 도구가 아닙니다**. 연구 및 교육 목적의 시각화 도구입니다.
- 실제 진단 및 치료는 반드시 **정신건강의학과 전문의**와 상담하시기 바랍니다.
- MDD는 이질성(heterogeneity)이 높은 질환으로, 개인별 경로는 크게 다를 수 있습니다.
"""

DEFAULT_MOODGARDEN_LIVE_PATH = "moodgarden_live.json"
ENABLE_LOCAL_LIVE_SYNC = os.getenv("ENABLE_LOCAL_LIVE_SYNC", "0") == "1"

EMA_CARD_CSS = """
<style>
.ema-card {
    background: rgba(28, 31, 54, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 8px;
    padding: 12px 14px;
    min-height: 86px;
}
.ema-card .label {
    color: rgba(255, 255, 255, 0.62);
    font-size: 0.78rem;
    line-height: 1.2;
    margin-bottom: 6px;
}
.ema-card .value {
    color: #ffffff;
    font-size: 1.32rem;
    font-weight: 700;
    line-height: 1.18;
    word-break: keep-all;
}
.ema-card .note {
    color: rgba(255, 255, 255, 0.70);
    font-size: 0.76rem;
    line-height: 1.25;
    margin-top: 5px;
}
.ema-insight {
    background: rgba(25, 28, 48, 0.96);
    border-left: 4px solid #70A1FF;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 0 0 8px 0;
}
.ema-insight.risk { border-left-color: #FF6B81; }
.ema-insight.protect { border-left-color: #2ED573; }
.ema-insight .title {
    color: #fff;
    font-weight: 700;
    font-size: 0.95rem;
    line-height: 1.25;
}
.ema-insight .body {
    color: rgba(255,255,255,0.75);
    font-size: 0.82rem;
    line-height: 1.35;
    margin-top: 4px;
}
</style>
"""

# ==================== 슬라이더 기본값 & 리셋 ====================

SLIDER_DEFAULTS = {
    # 유전 (슬라이더)
    "sl_prs": 30,
    "sl_nr3c1": 25,
    # 유전 (선택형 → 0/50/100 매핑)
    "sel_bdnf": "Val/Val (정상 범위)",
    "sel_fkbp5": "CC (정상)",
    "sel_httlpr": "La/La (정상)",
    # 뇌
    "sl_dmn": 40, "sl_sn": 35, "sl_sgacc": 35,
    "sl_reward": 35, "sl_amygdala": 40,
    "sl_pfc": 50, "sl_hippo": 60,
    # 생체
    "sl_cortisol": 35, "sl_inflammation": 25,
    # 치료
    "sl_antidepressant": 0,
}

GENO_OPTIONS = {
    "sel_bdnf": {
        "Val/Val (정상 범위)": 0,
        "Val/Met (중간 위험)": 50,
        "Met/Met (고위험)": 100,
    },
    "sel_fkbp5": {
        "CC (정상)": 0,
        "CT (보통)": 50,
        "TT (고위험, 특히 아동기 트라우마 시)": 100,
    },
    "sel_httlpr": {
        "La/La (정상)": 0,
        "La/S 또는 Lg/L (중간)": 50,
        "S/S 또는 Lg/S (논란, 효과 약함)": 80,
    },
}

# selectbox 기본값의 첫 번째 옵션 (리셋 시 사용)
GENO_DEFAULTS = {
    "sel_bdnf": "Val/Val (정상 범위)",
    "sel_fkbp5": "CC (정상)",
    "sel_httlpr": "La/La (정상)",
}

def reset_all():
    for k, v in SLIDER_DEFAULTS.items():
        st.session_state[k] = v


# ==================== 데이터 로드 ====================


def load_circuit_data():
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mdd_triangle_circuit_map.json"
    )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    circuit_data = load_circuit_data()
except FileNotFoundError:
    st.error("JSON 데이터 파일을 찾을 수 없습니다: mdd_triangle_circuit_map.json")
    st.stop()

# ==================== 시뮬레이션 함수 ====================


def logistic_phase_transition(x, threshold=0.5, steepness=5.0):
    return 1 / (1 + np.exp(-steepness * (x - threshold)))


def calculate_circuit_distortion(slider_values, circuit_name, use_nonlinear=True):
    try:
        info = circuit_data["simulation_model"]["circuit_distortion_functions"][
            circuit_name
        ]
        base = info["base_activity"]
        linear_sum = 0
        for factor, params in info["distortion_factors"].items():
            if factor in slider_values:
                w = params["weight"]
                d = params["direction"]
                v = slider_values[factor] / 100
                if d == "increase":
                    linear_sum += w * v
                elif d == "decrease":
                    linear_sum -= w * v
                elif d == "normalize":
                    linear_sum -= w * v
        if use_nonlinear:
            t = info.get("phase_transition_threshold", 0.5)
            s = info.get("phase_transition_steepness", 8.0)
            norm = np.clip(base + linear_sum, 0, 1)
            dist = logistic_phase_transition(norm, t, s)
            dist = base + (dist - 0.5) * 0.5
        else:
            dist = base + linear_sum
        return float(np.clip(dist, 0, 1))
    except KeyError:
        return 0.5


def predict_symptoms(distortions):
    preds = {}
    labels = {
        "emotional_symptoms": "🟥 정서 증상",
        "anhedonia_symptoms": "🟨 무쾌감",
        "cognitive_symptoms": "🟦 인지 저하",
        "rumination_symptoms": "🔄 반추",
        "somatic_symptoms": "💤 신체 증상",
        "social_symptoms": "👥 사회 위축",
    }
    for key, mapping in circuit_data["simulation_model"]["symptom_mapping"].items():
        vals = [distortions.get(c, 0.5) for c in mapping["primary_circuits"]]
        avg = float(np.mean(vals))
        preds[key] = {
            "label": labels.get(key, key),
            "severity": "높음" if avg > mapping["threshold_distortion"] else "낮음",
            "distortion": avg,
        }
    return preds


# ==================== 공통 유틸 ====================


def _edge_color(weight):
    """왜곡도(0~1) → 색상"""
    w = np.clip(weight, 0, 1)
    if w < 0.35:
        return f"rgba(100, 255, 100, 0.7)"
    elif w > 0.65:
        return f"rgba(255, 80, 80, 0.85)"
    else:
        r = int(100 + 155 * ((w - 0.35) / 0.30))
        g = int(255 - 175 * ((w - 0.35) / 0.30))
        return f"rgba({r}, {g}, 100, 0.75)"


def _add_line(fig, p1, p2, color, width, hover, dash=None):
    line_d = dict(width=width, color=color)
    if dash:
        line_d["dash"] = "dash"
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode="lines", line=line_d, hovertemplate=f"<b>{hover}</b><extra></extra>",
        showlegend=False,
    ))


def _curved_line(p1, p2, n=20, bow=0.15):
    """두 점 사이 곡선(베지에) 포인트 생성"""
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    mz = (p1[2] + p2[2]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    mx += -dy / ln * bow * ln
    my += dx / ln * bow * ln
    t = np.linspace(0, 1, n)
    xs = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * mx + t ** 2 * p2[0]
    ys = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * my + t ** 2 * p2[1]
    zs = (1 - t) ** 2 * p1[2] + 2 * (1 - t) * t * mz + t ** 2 * p2[2]
    return xs, ys, zs


# ==================== ① 단순화 삼각구조 ====================


def plot_simplified_triangle(slider_values, use_nonlinear=True):
    """
    유전(4) → 뇌(4) → 행동(4)  핵심 경로만 표시
    """
    fig = go.Figure()

    # --- 회로 왜곡도 ---
    ckeys = list(circuit_data["simulation_model"]["circuit_distortion_functions"].keys())
    dist = {k: calculate_circuit_distortion(slider_values, k, use_nonlinear)
            for k in ckeys}

    # --- 노드 정의 ---
    Z_GENE, Z_BRAIN, Z_BEH = 50, 0, -50

    genes = [
        ("BDNF Met\n(신경가소성↓)", -35, 15, "#FF6B6B", 15),
        ("FKBP5\n(스트레스↑)", -10, 22, "#FF8E72", 15),
        ("5-HTTLPR\n(세로토닌)", 15, 22, "#FFB347", 13),
        ("PRS\n(다유전자)", 38, 15, "#C44569", 13),
    ]
    brains = [
        ("DMN\n(반추 🔥)", -32, 18, "#FF4757", dist.get("dmn_overactivity", 0.3)),
        ("sgACC\n(우울고착 🎯)", -2, 28, "#FF6348", dist.get("sgacc_hyperactivity", 0.3)),
        ("보상\n(무쾌감 🍰)", 28, 8, "#FFD700", dist.get("reward_circuit_dysfunction", 0.3)),
        ("편도체\n(부정편향 🔔)", -18, -10, "#FF4444", dist.get("amygdala_hypersensitivity", 0.3)),
    ]
    behaviors = [
        ("우울감", -30, 18, "#E17055", 14),
        ("무쾌감", -5, 25, "#FDCB6E", 14),
        ("반추", 20, 25, "#E84393", 14),
        ("인지/사회↓", 40, 10, "#74B9FF", 13),
    ]

    # --- 층 배경 ---
    for zz, clr, lbl in [
        (Z_GENE, "rgba(255,100,100,0.03)", "🧬 유전 취약성"),
        (Z_BRAIN, "rgba(30,144,255,0.03)", "🧠 뇌 네트워크"),
        (Z_BEH, "rgba(162,155,254,0.03)", "😔 행동·증상"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[-50, 50, 50, -50], y=[-15, -15, 32, 32], z=[zz] * 4,
            mode="markers", marker=dict(size=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[0], y=[34], z=[zz + 4], mode="text", text=[lbl],
            textfont=dict(size=13, color="white"), showlegend=False, hoverinfo="skip",
        ))

    # --- 노드 그리기 ---
    for name, x, y, color, extra in genes:
        sz = extra
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[Z_GENE], mode="markers+text",
            marker=dict(size=sz, color=color, opacity=0.9, line=dict(color="white", width=0.5)),
            text=[name], textposition="top center", textfont=dict(size=9, color="white"),
            hovertemplate=f"<b>{name.replace(chr(10), ' ')}</b><br>유전 취약성<extra></extra>",
            showlegend=False,
        ))

    for name, x, y, color, d in brains:
        sz = 13 + d * 12
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[Z_BRAIN], mode="markers+text",
            marker=dict(size=sz, color=color, opacity=0.92, line=dict(color="white", width=0.5)),
            text=[name], textposition="top center", textfont=dict(size=9, color="white"),
            hovertemplate=f"<b>{name.replace(chr(10), ' ')}</b><br>왜곡도: {d:.2f}<extra></extra>",
            showlegend=False,
        ))

    for name, x, y, color, sz in behaviors:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[Z_BEH], mode="markers+text",
            marker=dict(size=sz, color=color, opacity=0.9, line=dict(color="white", width=0.5)),
            text=[name], textposition="bottom center", textfont=dict(size=9, color="white"),
            hovertemplate=f"<b>{name}</b><br>행동·증상<extra></extra>",
            showlegend=False,
        ))

    # --- 연결선: 유전→뇐 (4개) ---
    g2b = [
        (0, 0, "BDNF↓ → DMN 과활성", 0.6),
        (1, 1, "FKBP5↑ → sgACC 과활성", 0.7),
        (2, 3, "5-HTTLPR → 편도체 과민", 0.5),
        (3, 2, "PRS → 보상 회로 이상", 0.4),
    ]
    for gi, bi, lbl, bw in g2b:
        gn, gx, gy, gc, gs = genes[gi]
        bn, bx, by, bc, bd = brains[bi]
        w = bw * (0.5 + bd / 2)
        c = _edge_color(w)
        lw = max(3, w * 10)
        _add_line(fig, (gx, gy, Z_GENE), (bx, by, Z_BRAIN), c, lw, lbl)

    # --- 연결선: 뇌↔뇐 (4개) ---
    b2b = [
        (0, 1, "DMN ↔ sgACC 과연결", 0.7, dist.get("dmn_overactivity", 0.3),
         dist.get("sgacc_hyperactivity", 0.3)),
        (1, 3, "sgACC → 편도체 증폭", 0.6, dist.get("sgacc_hyperactivity", 0.3),
         dist.get("amygdala_hypersensitivity", 0.3)),
        (0, 2, "DMN ↔ 보상 (반추-무쾌감)", 0.4, dist.get("dmn_overactivity", 0.3),
         dist.get("reward_circuit_dysfunction", 0.3)),
        (3, 0, "편도체 → DMN (부정→반추)", 0.4, dist.get("amygdala_hypersensitivity", 0.3),
         dist.get("dmn_overactivity", 0.3)),
    ]
    for si, ti, lbl, bw, sd, td in b2b:
        sn, sx, sy, sc, _ = brains[si]
        tn, tx, ty, tc, _ = brains[ti]
        w = bw * (0.5 + (sd + td) / 4)
        c = _edge_color(w)
        lw = max(3, w * 10)
        _add_line(fig, (sx, sy, Z_BRAIN), (tx, ty, Z_BRAIN), c, lw, lbl)

    # --- 연결선: 뇌→행동 (4개) ---
    b2v = [
        (0, 2, "DMN → 반추", 0.7),
        (1, 0, "sgACC → 우울감", 0.7),
        (2, 1, "보상↓ → 무쾌감", 0.7),
        (3, 3, "편도체 → 인지/사회↓", 0.5),
    ]
    for bi, vi, lbl, bw in b2v:
        bn, bx, by, bc, bd = brains[bi]
        vn, vx, vy, vc, vs = behaviors[vi]
        w = bw * (0.5 + bd / 2)
        c = _edge_color(w)
        lw = max(3, w * 10)
        _add_line(fig, (bx, by, Z_BRAIN), (vx, vy, Z_BEH), c, lw, lbl)

    # --- 피드백: 행동→유전 (1개, 점선) ---
    fkbp5 = genes[1]
    sleep_beh = behaviors[0]  # 우울감 → FKBP5 후성유전
    _add_line(fig,
              (sleep_beh[1], sleep_beh[2], Z_BEH),
              (fkbp5[1], fkbp5[2], Z_GENE),
              "rgba(255,255,100,0.35)", 2.5,
              "🔄 후성유전 피드백", dash=True)

    # --- 레이아웃 ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, range=[-55, 55],
                       title=""),
            yaxis=dict(showgrid=False, showticklabels=False, range=[-20, 40],
                       title=""),
            zaxis=dict(showgrid=False, showticklabels=False, range=[-60, 60],
                       title=""),
            bgcolor="rgba(15, 15, 35, 0.95)",
            camera=dict(eye=dict(x=0.0, y=-2.0, z=0.5),
                        center=dict(x=0, y=0, z=0)),
        ),
        paper_bgcolor="rgba(15, 15, 35, 1)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )
    return fig, dist


# ==================== ② 뇌 모양 회로도 ====================

# 뇌 내 영역 좌표 (근사 MNI)
BRAIN_REGIONS = {
    "mPFC":        {"x":   0, "y":  55, "z": -2,  "color": "#FF4757",
                    "label": "mPFC (DMN)", "dist_key": "dmn_overactivity"},
    "PCC":         {"x":   0, "y": -55, "z": 28,  "color": "#FF6B6B",
                    "label": "PCC (DMN)", "dist_key": "dmn_overactivity"},
    "sgACC":       {"x":   8, "y":  18, "z": -12, "color": "#FF6348",
                    "label": "sgACC (BA25)", "dist_key": "sgacc_hyperactivity"},
    "DLPFC":       {"x": -42, "y":  38, "z": 28,  "color": "#1E90FF",
                    "label": "DLPFC (CEN)", "dist_key": "pfc_dysregulation"},
    "dACC":        {"x":   2, "y":  32, "z": 28,  "color": "#3742FA",
                    "label": "dACC (SN/CEN)", "dist_key": "salience_network_dysfunction"},
    "AntInsula":   {"x": -38, "y":  22, "z": -2,  "color": "#FFA502",
                    "label": "전섬엽 (SN)", "dist_key": "salience_network_dysfunction"},
    "Amygdala":    {"x": -24, "y":  -2, "z": -18, "color": "#FF4444",
                    "label": "편도체", "dist_key": "amygdala_hypersensitivity"},
    "NAc":         {"x":  12, "y":  14, "z": -6,  "color": "#FFD700",
                    "label": "보상 (NAc)", "dist_key": "reward_circuit_dysfunction"},
    "Hippocampus": {"x": -28, "y": -18, "z": -16, "color": "#A29BFE",
                    "label": "해마", "dist_key": "hippocampus_atrophy"},
    "Hypothalamus":{"x":   2, "y":  -4, "z": -8,  "color": "#FD79A8",
                    "label": "시상하부 (HPA)", "dist_key": "hpa_axis_dysregulation"},
}

BRAIN_CONNECTIONS = [
    # (source, target, type, base_weight, description)
    ("mPFC",   "sgACC",      "over",    0.7, "DMN-sgACC 과연결 → 우울 고착"),
    ("mPFC",   "PCC",        "over",    0.6, "DMN 내부 과활성 → 반추"),
    ("sgACC",  "Amygdala",   "over",    0.6, "정서 증폭 루프"),
    ("DLPFC",  "Amygdala",   "under",   0.5, "탑다운 억제 ↓"),
    ("AntInsula", "dACC",    "dysreg",  0.5, "SN 네트워크 조절 장애"),
    ("DLPFC",  "dACC",       "dysreg",  0.4, "CEN 전환 실패"),
    ("NAc",    "DLPFC",      "under",   0.4, "보상-인지 연결 ↓"),
    ("Hippocampus", "Amygdala", "over", 0.5, "기억-정서 과결합"),
    ("Hypothalamus", "Hippocampus", "over", 0.5, "HPA→해마 (코르티솔 독성)"),
    ("mPFC",   "AntInsula",  "dysreg",  0.4, "DMN↔SN 전환 장애"),
]


def _brain_wireframe():
    """반투명 뇌 외곽선 (와이어프레임)"""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)

    X = 68 * np.sin(V) * np.cos(U)
    Y = 82 * np.sin(V) * np.sin(U)
    Z = 58 * np.cos(V) + 5

    # 위쪽 약간 넓게
    w_mod = 1.0 + 0.08 * np.clip((Z - 5) / 58, 0, 1)
    X = X * w_mod

    # 아래쪽 약간 납작하게
    flat = Z < -10
    Z[flat] = -10 + (Z[flat] + 10) * 0.75

    return X, Y, Z


def plot_brain_circuit(slider_values, use_nonlinear=True):
    """뇌 모양 내부에 회로 노드+연결 표시"""
    fig = go.Figure()

    # 왜곡도 계산
    ckeys = list(circuit_data["simulation_model"]["circuit_distortion_functions"].keys())
    dist = {k: calculate_circuit_distortion(slider_values, k, use_nonlinear)
            for k in ckeys}

    # --- 뇌 외곽선 ---
    BX, BY, BZ = _brain_wireframe()

    # 위도선 (5개)
    lat_x, lat_y, lat_z = [], [], []
    for i in range(0, BX.shape[0], BX.shape[0] // 6):
        lat_x.extend(list(BX[i, :]) + [None])
        lat_y.extend(list(BY[i, :]) + [None])
        lat_z.extend(list(BZ[i, :]) + [None])
    fig.add_trace(go.Scatter3d(
        x=lat_x, y=lat_y, z=lat_z, mode="lines",
        line=dict(color="rgba(130,150,200,0.12)", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    # 경도선 (8개)
    lon_x, lon_y, lon_z = [], [], []
    for j in range(0, BX.shape[1], BX.shape[1] // 9):
        lon_x.extend(list(BX[:, j]) + [None])
        lon_y.extend(list(BY[:, j]) + [None])
        lon_z.extend(list(BZ[:, j]) + [None])
    fig.add_trace(go.Scatter3d(
        x=lon_x, y=lon_y, z=lon_z, mode="lines",
        line=dict(color="rgba(130,150,200,0.12)", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    # --- 노드 ---
    for name, r in BRAIN_REGIONS.items():
        d = dist.get(r["dist_key"], 0.3)
        sz = 10 + d * 10
        fig.add_trace(go.Scatter3d(
            x=[r["x"]], y=[r["y"]], z=[r["z"]],
            mode="markers+text",
            marker=dict(size=sz, color=r["color"], opacity=0.92,
                        line=dict(color="white", width=1)),
            text=[r["label"]],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            hovertemplate=(
                f"<b>{r['label']}</b><br>"
                f"왜곡도: {d:.3f}<br>"
                f"MNI: ({r['x']}, {r['y']}, {r['z']})<extra></extra>"
            ),
            showlegend=False,
        ))

    # --- 연결선 ---
    type_styles = {
        "over":   ("🔴 과활성", lambda w: f"rgba(255,80,80,{0.4+0.5*w})"),
        "under":  ("🔵 저활성", lambda w: f"rgba(80,150,255,{0.4+0.5*w})"),
        "dysreg": ("🟠 조절장애", lambda w: f"rgba(255,180,50,{0.4+0.4*w})"),
    }
    for src, tgt, ctype, bw, desc in BRAIN_CONNECTIONS:
        s = BRAIN_REGIONS[src]
        t = BRAIN_REGIONS[tgt]
        sd = dist.get(s["dist_key"], 0.3)
        td = dist.get(t["dist_key"], 0.3)
        w = bw * (0.5 + (sd + td) / 4)
        style_name, color_fn = type_styles[ctype]
        color = color_fn(w)
        lw = max(2.5, w * 8)

        # 곡선 연결선
        cx, cy, cz = _curved_line(
            (s["x"], s["y"], s["z"]),
            (t["x"], t["y"], t["z"]),
            n=25, bow=0.08
        )
        dash_str = "dash" if ctype == "dysreg" else None
        line_d = dict(width=lw, color=color)
        if dash_str:
            line_d["dash"] = "dash"
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode="lines",
            line=line_d,
            hovertemplate=f"<b>{s['label']} ↔ {t['label']}</b><br>"
                          f"{desc}<br>유형: {style_name}<extra></extra>",
            showlegend=False,
        ))

    # --- 네트워크 범례 ( annotation ) ---
    legend_items = [
        ("🔴 과활성 연결", "rgba(255,80,80,0.8)"),
        ("🔵 저활성 연결", "rgba(80,150,255,0.8)"),
        ("🟠 조절장애", "rgba(255,180,50,0.8)"),
    ]

    # --- 레이아웃 ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, range=[-80, 80],
                       title=""),
            yaxis=dict(showgrid=False, showticklabels=False, range=[-90, 70],
                       title=""),
            zaxis=dict(showgrid=False, showticklabels=False, range=[-70, 70],
                       title=""),
            bgcolor="rgba(15, 15, 35, 0.95)",
            camera=dict(eye=dict(x=0.8, y=-1.6, z=0.7),
                        center=dict(x=0, y=-0.05, z=-0.05)),
            aspectmode="data",
        ),
        paper_bgcolor="rgba(15, 15, 35, 1)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650,
        title=dict(text="🧠 MDD 뇌 회로 지도 — 드래그하여 회전하세요",
                   font=dict(size=14, color="rgba(255,255,255,0.7)")),
    )
    return fig, dist


# ==================== 상전이 곡선 ====================

NAME_MAP = {
    "dmn_overactivity": "DMN 과활성 (반추)",
    "salience_network_dysfunction": "SN 조절장애",
    "sgacc_hyperactivity": "sgACC 과활성 (우울 고착)",
    "reward_circuit_dysfunction": "보상 회로 이상 (무쾌감)",
    "amygdala_hypersensitivity": "편도체 과민",
    "pfc_dysregulation": "PFC 조절 이상",
    "hippocampus_atrophy": "해마 위축",
    "hpa_axis_dysregulation": "HPA 축/염증",
}


def plot_phase_transition(circuit_name, ema_df=None):
    x_vals = np.linspace(0, 1, 120)

    def _make_vals(x):
        return {k: x * 100 for k in circuit_data["simulation_model"]["variables"]}

    lin_y = [calculate_circuit_distortion(_make_vals(x), circuit_name, False)
             for x in x_vals]
    non_y = [calculate_circuit_distortion(_make_vals(x), circuit_name, True)
             for x in x_vals]

    info = circuit_data["simulation_model"]["circuit_distortion_functions"][circuit_name]
    threshold = info.get("phase_transition_threshold", 0.5)

    fig = go.Figure()

    # 영역 채우기: 정상(좌) / 임계(중간) / 병리(우)
    fig.add_shape(type="rect", x0=0, y0=0, x1=threshold - 0.05, y1=1,
                  fillcolor="rgba(46,213,115,0.06)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=threshold - 0.05, y0=0, x1=threshold + 0.05, y1=1,
                  fillcolor="rgba(255,165,2,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=threshold + 0.05, y0=0, x1=1, y1=1,
                  fillcolor="rgba(255,71,87,0.06)", line_width=0, layer="below")

    # 구간 라벨
    fig.add_annotation(x=(0 + threshold - 0.05) / 2, y=0.95,
                       text="✅ 정상 구간", showarrow=False,
                       font=dict(size=13, color="#2ED573", family="Arial Black"))
    fig.add_annotation(x=threshold, y=0.95,
                       text="⚡임계", showarrow=False,
                       font=dict(size=12, color="#FFA502", family="Arial Black"))
    fig.add_annotation(x=(threshold + 0.05 + 1) / 2, y=0.95,
                       text="🔴 병리 구간", showarrow=False,
                       font=dict(size=13, color="#FF4757", family="Arial Black"))

    # 초록 임계선
    fig.add_shape(type="line", x0=threshold, y0=0, x1=threshold, y1=1,
                  line=dict(color="#2ED573", width=2.5, dash="dot"))

    # 파란 점선: 단순 선형 모델
    fig.add_trace(go.Scatter(
        x=x_vals, y=lin_y, mode="lines",
        name="🔵 선형 모델 (점진적 변화)",
        line=dict(color="#5B9BFF", width=2.5, dash="dash"),
        hovertemplate="선형: %{y:.3f}<extra></extra>",
    ))

    # 빨간 실선: 비선형 상전이
    fig.add_trace(go.Scatter(
        x=x_vals, y=non_y, mode="lines",
        name="🔴 비선형 상전이 (급격한 전환)",
        line=dict(color="#FF4757", width=3.5),
        hovertemplate="비선형: %{y:.3f}<extra></extra>",
    ))

    # 임계점 마커
    idx_t = int(threshold * 120)
    idx_t = min(idx_t, len(non_y) - 1)
    fig.add_trace(go.Scatter(
        x=[threshold], y=[non_y[idx_t]], mode="markers+text",
        name="🟢 임계점 (Critical Point)",
        marker=dict(size=12, color="#2ED573", symbol="diamond",
                    line=dict(color="white", width=2)),
        text=[f"  임계점 = {threshold:.2f}"],
        textposition="middle right",
        textfont=dict(size=13, color="#2ED573", family="Arial Black"),
        hovertemplate=f"임계점: {threshold:.2f}<extra></extra>",
    ))

    if ema_df is not None:
        fig = add_ema_phase_overlay(fig, ema_df, circuit_name)

    fig.update_layout(
        title=dict(
            text=f"{NAME_MAP.get(circuit_name, circuit_name)} — 상전이 곡선",
            font=dict(size=16, color="white"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.12,
            xanchor="center", x=0.5,
            font=dict(size=13, color="white"),
            bgcolor="rgba(30,30,60,0.8)",
            bordercolor="rgba(100,100,100,0.5)", borderwidth=1,
        ),
        xaxis_title=dict(text="요인 종합 값 (유전+환경+뇌 상태)", font=dict(size=13)),
        yaxis_title=dict(text="회로 왜곡도 (0=정상, 1=병리)", font=dict(size=13)),
        height=500, hovermode="x unified",
        paper_bgcolor="rgba(15,15,35,1)",
        font=dict(color="white", size=12),
        plot_bgcolor="rgba(20,20,40,1)",
        margin=dict(t=100),
        xaxis=dict(
            gridcolor="rgba(100,100,100,0.2)",
            tickfont=dict(size=12),
            dtick=0.1,
        ),
        yaxis=dict(
            gridcolor="rgba(100,100,100,0.2)",
            tickfont=dict(size=12),
        ),
    )
    return fig


# ==================== 증상 레이더 ====================


def plot_symptom_radar(symptom_predictions):
    labels = [v["label"] for v in symptom_predictions.values()]
    values = [v["distortion"] for v in symptom_predictions.values()]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(255,71,87,0.25)",
        line=dict(color="#FF4757", width=2),
        marker=dict(size=8, color=["red" if v["severity"] == "높음" else "green"
                                    for v in symptom_predictions.values()]),
        name="현재 상태",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor="rgba(100,100,100,0.3)"),
                   bgcolor="rgba(15,15,35,1)"),
        height=400, paper_bgcolor="rgba(15,15,35,1)", font=dict(color="white"),
    )
    return fig


# ==================== 메인 ====================


def main():
    st.title("🧠 주요 우울장애(MDD) 유전-뇌-행동 삼각구조")
    st.markdown(WARNING_TEXT)

    # ---------- 사이드바 ----------
    st.sidebar.title("⚙️ 시뮬레이션 조절")

    st.sidebar.subheader("🔬 모델 설정")
    use_nonlinear = st.sidebar.checkbox(
        "비선형 상전이 모델 사용", value=True,
        help="Yuan 2026 Neuron — 임계점 돌파 시 급격한 변화",
    )
    st.sidebar.button("🔄 기본값으로 리셋", on_click=reset_all)

    st.sidebar.subheader("🌱 MoodGarden EMA")
    ema_file = st.sidebar.file_uploader(
        "MoodGarden CSV 업로드",
        type=["csv"],
        help="MoodGarden의 CSV 내보내기 파일을 업로드하면 최근 EMA가 회로 슬라이더에 반영됩니다.",
    )
    live_sync = False
    live_path = DEFAULT_MOODGARDEN_LIVE_PATH
    if ENABLE_LOCAL_LIVE_SYNC:
        live_sync = st.sidebar.checkbox(
            "MoodGarden 실시간 파일 동기화",
            value=False,
            help="MoodGarden Electron 앱이 쓰는 moodgarden_live.json 파일을 읽어 자동 갱신합니다. 로컬 실행 전용입니다.",
        )
        live_path = st.sidebar.text_input(
            "실시간 JSON 경로",
            value=DEFAULT_MOODGARDEN_LIVE_PATH,
            disabled=not live_sync,
        )
    else:
        st.sidebar.caption("외부 공개 모드에서는 MoodGarden CSV 업로드로 EMA를 반영합니다.")
    ema_days = st.sidebar.slider("EMA 반영 기간", 3, 30, 7, key="ema_days")
    apply_ema = st.sidebar.checkbox("EMA 데이터로 슬라이더 자동 설정", value=False)

    ema_df = None
    ema_summary = None

    if live_sync:
        if os.path.exists(live_path):
            try:
                ema_df = parse_moodgarden_json_file(live_path)
                if ema_df.empty:
                    fallback_csv = latest_moodgarden_csv(os.path.dirname(live_path))
                    if fallback_csv:
                        with open(fallback_csv, "rb") as f:
                            ema_df = parse_moodgarden_csv(f)
                        st.session_state["_ema_live_fallback_csv"] = fallback_csv
                        st.session_state["_ema_live_fallback_mtime"] = os.path.getmtime(fallback_csv)
                    else:
                        st.session_state.pop("_ema_live_fallback_csv", None)
                        st.session_state.pop("_ema_live_fallback_mtime", None)
                else:
                    st.session_state.pop("_ema_live_fallback_csv", None)
                    st.session_state.pop("_ema_live_fallback_mtime", None)
                ema_summary = summarize_ema(ema_df, ema_days)
                st.session_state["ema_df"] = ema_df
                st.session_state["ema_summary"] = ema_summary
                st.session_state["_ema_live_mtime"] = os.path.getmtime(live_path)
                fallback_label = ""
                if st.session_state.get("_ema_live_fallback_csv"):
                    fallback_label = " · 최신 CSV fallback"
                st.sidebar.success(
                    f"실시간 동기화: {ema_summary.total_checkins}개 체크인 · {ema_summary.unique_days}일{fallback_label}"
                )
            except Exception as exc:
                st.sidebar.error(f"실시간 JSON을 읽지 못했습니다: {exc}")
        else:
            st.sidebar.warning("실시간 JSON 파일이 아직 없습니다. MoodGarden Electron 앱을 열어주세요.")
    elif ema_file is not None:
        try:
            ema_df = parse_moodgarden_csv(ema_file)
            ema_summary = summarize_ema(ema_df, ema_days)
            st.session_state["ema_df"] = ema_df
            st.session_state["ema_summary"] = ema_summary
            st.sidebar.success(
                f"{ema_summary.total_checkins}개 체크인 · {ema_summary.unique_days}일 로드"
            )
        except Exception as exc:
            st.sidebar.error(f"EMA CSV를 읽지 못했습니다: {exc}")
            ema_df = None
            ema_summary = None
    elif "ema_df" in st.session_state:
        ema_df = st.session_state["ema_df"]
        ema_summary = summarize_ema(ema_df, ema_days)
        st.session_state["ema_summary"] = ema_summary

    if live_sync and hasattr(st, "fragment"):
        @st.fragment(run_every="5s")
        def watch_moodgarden_live_file():
            if not os.path.exists(live_path):
                return
            mtime = os.path.getmtime(live_path)
            previous = st.session_state.get("_ema_live_mtime")
            if previous is not None and mtime != previous:
                st.session_state["_ema_live_mtime"] = mtime
                st.rerun()
            fallback_csv = st.session_state.get("_ema_live_fallback_csv")
            if fallback_csv and os.path.exists(fallback_csv):
                fallback_mtime = os.path.getmtime(fallback_csv)
                previous_fallback = st.session_state.get("_ema_live_fallback_mtime")
                if previous_fallback is not None and fallback_mtime != previous_fallback:
                    st.session_state["_ema_live_fallback_mtime"] = fallback_mtime
                    st.rerun()

        watch_moodgarden_live_file()
        st.sidebar.caption("실시간 파일 감시 중 · 5초 간격")

    if apply_ema and ema_df is not None:
        ema_slider_values = ema_to_slider_values(ema_df, ema_days)
        for key, value in ema_slider_values.items():
            if key not in st.session_state or st.session_state.get("_ema_last_applied") != ema_slider_values:
                st.session_state[key] = value
        st.session_state["_ema_last_applied"] = ema_slider_values
        st.sidebar.caption("EMA 기반 추정값이 뇌/생체/치료 슬라이더에 적용되었습니다.")
    elif ema_df is not None:
        st.sidebar.caption("CSV가 로드되었습니다. 체크하면 슬라이더에 반영됩니다.")

    # ========== 유전 계층 ==========
    st.sidebar.subheader("🧬 유전적 취약성")

    st.sidebar.markdown("🔸 **PRS (다유전 점수)** 🟢 *연속형*")
    prs = st.sidebar.slider("PRS 백분위", 0, 100, 30, key="sl_prs",
                             help="🟢 다유전 점수(PRS) = 수백 개 SNP의 가중합 → 정규분포 → 0-100 퍼센타일 가능\nPGC MDD 2025 Cell: MDD PRS 설명 분산 ~8.9%\n확립된 지표로, 0-100 슬라이더가 과학적으로 타당")

    st.sidebar.markdown("🔸 **BDNF Val66Met** 🔴 *범주형 (유전형)*")
    bdnf_sel = st.sidebar.selectbox(
        "BDNF 유전형 선택", list(GENO_OPTIONS["sel_bdnf"].keys()),
        key="sel_bdnf",
        help="🔴 유전형은 범주형(Val/Val, Val/Met, Met/Met)이므로 0-100 슬라이더가 부정확\n"
             "BDNF Met 대립유전자: 우울증 OR≈1.2-1.5 (소효과)\n"
             "단백질 수준(ELISA)은 연속형이나, 여기서는 유전형만 표시")
    bdnf = GENO_OPTIONS["sel_bdnf"][bdnf_sel]

    st.sidebar.markdown("🔸 **FKBP5 rs1360780** 🔴 *범주형*")
    fkbp5_sel = st.sidebar.selectbox(
        "FKBP5 유전형 선택", list(GENO_OPTIONS["sel_fkbp5"].keys()),
        key="sel_fkbp5",
        help="🔴 주효과 OR=1.062 (거의 의미 없음)\n"
             "G×E(아동기 트라우마×T 대립유전자)에서만 의미 (OR 1.5-2.0)\n"
             "트라우마 없으면 단독 효과 제로")
    fkbp5 = GENO_OPTIONS["sel_fkbp5"][fkbp5_sel]

    st.sidebar.markdown("🔸 **5-HTTLPR** 🔴 *범주형·논란*")
    httrlpr_sel = st.sidebar.selectbox(
        "5-HTTLPR 유전형 선택", list(GENO_OPTIONS["sel_httlpr"].keys()),
        key="sel_httlpr",
        help="🔴 Risch et al. (2009, JAMA, n=14,250)에서 stress×5-HTTLPR 유의효과 없음\n"
             "대규모 메타분석에서 재현 실패. 소규모 연구만 작은 효과 보고\n"
             "가장 논란이 많은 G×E 후보")
    httrlpr = GENO_OPTIONS["sel_httlpr"][httrlpr_sel]

    st.sidebar.markdown("🔸 **NR3C1 메틸화** 🟡 *연속형 (메틸화%)*")
    nr3c1 = st.sidebar.slider("NR3C1 메틸화 수준", 0, 100, 25, key="sl_nr3c1",
                               help="🟡 NR3C1(당질코르티코이드 수용체) 메틸화 수준 0-100%\n"
                                    "메틸화 1%↑ → 우울 가능성 9%↑ (총효과: 소-중간)\n"
                                    "한계: 조직 특이성 (혈≠뇌), 나이/약물 혼란 변수\n"
                                    "공식: Score = min(100, NR3C1_methylation% × 5)")

    # ========== 뇌 회로 계층 ==========
    st.sidebar.subheader("🧠 뇌 네트워크 상태")
    st.sidebar.caption("ⓘ fMRI 기반 추정치 — 임상에서 직접 측정 어려움")

    dmn_val = st.sidebar.slider("🟡 DMN 과활성 (반추)", 0, 100, 40, key="sl_dmn",
                                 help="🟡 fMRI 안정 시 DMN 기능연결성 (Fisher Z) → z-score → 백분위\n"
                                      "Cohen's d ~0.4-0.8. PCC-mPFC 과연결이 핵심\n"
                                      "한계: 높은 개인간 변이, 불안/PTSD에서도 나타남 (비특이적)")
    sn_val = st.sidebar.slider("🟡 SN 조절장애", 0, 100, 35, key="sl_sn",
                                help="🟡 Sacchet et al. (2024, Nature): SN이 우울증에서 피질 ~2배 확장\n"
                                     "정상: 피질 면적의 5-8%, 우울: 10-16%\n"
                                     "한계: 2024년 발견, 복제 진행 중")
    sgacc_val = st.sidebar.slider("🟡 sgACC 과활성", 0, 100, 35, key="sl_sgacc",
                                   help="🟡 sgACC(MNI ±5,20,-10) BOLD %변화율 → z-score\n"
                                        "Deep TMS 타겟. 부정 정서 처리 시 과활성\n"
                                        "한계: 작은 영역, 정밀 위치 필요, 스캐너 간 높은 변이")
    reward_val = st.sidebar.slider("🟡 보상 회로 저기능", 0, 100, 35, key="sl_reward",
                                    help="🟡 복부 줄기질(VS) BOLD. Ng 2018 메타: d~0.3-0.5\n"
                                         "무쾌감 ≈ 동기 장애 (Treadway 2012)\n"
                                         "EMA anhedonia 슬라이더와 r≈-0.3~-0.5 상관\n"
                                         "한계: 과제 의존적, 현재 기분에 의존")
    amyg_val = st.sidebar.slider("🟡 편도체 감도", 0, 100, 40, key="sl_amygdala",
                                  help="🟡 부정 자극에 과반응. d~0.4-0.6. SSRI로 감소\n"
                                       "BOLD 0.2-0.8% 범위\n"
                                       "한계: 자극 의존적, 약물 효과, 높은 개인 변이")
    pfc_val = st.sidebar.slider("🟡 전두엽 조절 (↑=건강)", 0, 100, 50, key="sl_pfc",
                                 help="🟡 dlPFC/vlPFC 활성도. 인지조절 점수와 상관\n"
                                      "대안: 실행기능 검사 T-점호 직접 사용 가능\n"
                                      "한계: 다중 하위영역, 과제 의존적")
    hippo_val = st.sidebar.slider("🟢 해마 체적 (↑=건강)", 0, 100, 60, key="sl_hippo",
                                   help="🟢 구조 MRI + FreeSurfer 자동 체적 측정\n"
                                        "UK Biobank N>19,700 노멋티브 데이터 확보\n"
                                        "양측 해마 ~3,800-4,000 mm³ (청년). MDD에서 3-15% 감소\n"
                                        "공식: Score = ((volume-2600)/2600)×100\n"
                                        "한계: 연령/성별 효과, 스캐너 의존적")

    # ========== 생체 지표 계층 ==========
    st.sidebar.subheader("🌡️ 생체 지표")
    st.sidebar.caption("ⓘ 실제 혈액/타액 검사로 측정 가능")

    cortisol_val = st.sidebar.slider("🟢 코르티솔 수준", 0, 100, 35, key="sl_cortisol",
                                      help="🟢 타액 코르티솔 (가장 일반적). 확립된 정상 범위 존재\n"
                                           "아침 타액: 8-20 nmol/L, 취침 전: <4 nmol/L\n"
                                           "우울: 과코르티솔(전형) + 저코르티솔(비정형) 모두 가능\n"
                                           "공식: Score = (cortisol_8am_μg/dL / 25) × 100\n"
                                           "한계: 일주기 변화 → 시간 지정 채혈 필수")
    inflam_val = st.sidebar.slider("🟢 염증 수준 (CRP/IL-6)", 0, 100, 25,
                                    key="sl_inflammation",
                                    help="🟢 고감도 CRP + IL-6 ELISA. AHA/CDC 컷오프 확립\n"
                                         "CRP: <1 저위험, 1-3 중간, >3 고위험 (mg/L)\n"
                                         "IL-6: 건강 중앙값 ~1.5 pg/mL, 우울 컷오프 ~3.77\n"
                                         "Kappelmann 2026 JAMA Psychiatry: IL-6 인과적 역할 지지\n"
                                         "공식: Score = min(100, CRP×10 + IL6×8)\n"
                                         "한계: 급성 감염 혼란, BMI와 CRP 강한 상관")

    # ========== 치료 ==========
    st.sidebar.subheader("💊 치료 효과")
    ad_val = st.sidebar.slider("항우울제 반응도", 0, 100, 0, key="sl_antidepressant")

    # --- 선택형 → 수치 매핑 ---
    crhr1 = 30  # CRHR1은 보호 변이로 제거됐으나 시뮬레이션 호환성 위해 기본값 유지

    slider_values = {
        "bdnf_expression": bdnf, "fkbp5_expression": fkbp5,
        "crhr1_sensitivity": crhr1, "serotonin_transporter": httrlpr,
        "nr3c1_dysfunction": nr3c1,
        "dmn_overactivity": dmn_val, "salience_dysfunction": sn_val,
        "sgacc_hyperactivity": sgacc_val, "reward_hypofunction": reward_val,
        "amygdala_sensitivity": amyg_val, "prefrontal_control": pfc_val,
        "hippocampal_volume": hippo_val,
        "cortisol_level": cortisol_val, "inflammation_level": inflam_val,
        "antidepressant_response": ad_val,
    }

    # ---------- 탭 ----------
    tab1, tab_ema, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ 증상 프로파일",
        "🌱 EMA 대시보드",
        "2️⃣ 삼각구조 (개요)",
        "3️⃣ 뇌 회로 지도 (3D)",
        "4️⃣ 다차원 원인",
        "5️⃣ 치료 가이드",
        "📖 사용 설명서",
    ])

    # ===== Tab 1: 증상 프로파일 =====
    with tab1:
        st.header("📊 증상 프로파일 시뮬레이션")
        st.markdown("사이드바 슬라이더를 조절하면 증상 심각도가 실시간으로 변합니다.")

        ckeys = list(circuit_data["simulation_model"]["circuit_distortion_functions"].keys())
        distortions = {k: calculate_circuit_distortion(slider_values, k, use_nonlinear)
                       for k in ckeys}
        symptom_pred = predict_symptoms(distortions)

        col_r, col_l = st.columns([1, 1])
        with col_r:
            st.plotly_chart(plot_symptom_radar(symptom_pred), width='stretch')
        with col_l:
            for key, result in symptom_pred.items():
                icon = "🔴" if result["severity"] == "높음" else "🟢"
                st.markdown(f"**{icon} {result['label']}**")
                st.progress(min(result["distortion"], 1.0))
                st.caption(f"왜곡도: {result['distortion']:.3f} | 심각도: {result['severity']}")
                st.markdown("")

        st.subheader("📋 DSM-5 MDD 진단 기준 (참고용)")
        st.markdown("최소 2주 동안, 다음 중 **5개 이상** 해당:")
        dsm = [
            ("① 우울한 기분", "거의 매일"),
            ("② 흥미/즐거움 상실", "무쾌감"),
            ("③ 체중/식욕 변화", "±5%"),
            ("④ 수면 장애", "불면/과수면"),
            ("⑤ 정신운동 변화", "초조/지체"),
            ("⑥ 피로/에너지 상실", "매일"),
            ("⑦ 무가치감/죄책감", "과도함"),
            ("⑧ 집중력 저하", "매일"),
            ("⑨ 자살 생각", "반복적"),
        ]
        cols9 = st.columns(3)
        for i, (name, desc) in enumerate(dsm):
            with cols9[i % 3]:
                st.checkbox(name, key=f"dsm_{i}", help=desc)

    # ===== EMA Dashboard =====
    with tab_ema:
        st.header("🌱 MoodGarden EMA 대시보드")
        if ema_df is None or ema_summary is None:
            st.info("사이드바에서 MoodGarden CSV를 업로드하면 일상 기분 궤적과 회로 추정값이 표시됩니다.")
        else:
            st.markdown(EMA_CARD_CSS, unsafe_allow_html=True)
            st.markdown(
                """
                MoodGarden의 EMA 값은 직접적인 뇌 측정값이 아니라, 문헌 기반 회로 모델을 움직이는
                **행동·경험 앵커**로 사용됩니다.
                """
            )

            warning_note = ema_summary.warning_label
            if len(warning_note) > 16:
                warning_value = "주의 필요" if warning_note != "뚜렷한 조기 경고 없음" else "안정"
            else:
                warning_value = warning_note

            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.15])
            with c1:
                st.markdown(
                    f"""
                    <div class="ema-card">
                      <div class="label">최근 MDD 위험 지수</div>
                      <div class="value">{ema_summary.risk_score:.1f}/100</div>
                      <div class="note">최근 {ema_summary.recent_days}일 평균</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="ema-card">
                      <div class="label">기록량</div>
                      <div class="value">{ema_summary.total_checkins}개</div>
                      <div class="note">{ema_summary.unique_days}일 · 하루 {ema_summary.average_per_day:.1f}회</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="ema-card">
                      <div class="label">최근 추세</div>
                      <div class="value">{ema_summary.trend_label}</div>
                      <div class="note">일별 위험 지수 기울기</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""
                    <div class="ema-card">
                      <div class="label">조기 경고</div>
                      <div class="value">{warning_value}</div>
                      <div class="note">{warning_note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("최근 흐름")
            st.plotly_chart(plot_risk_timeline(ema_df), width="stretch")

            mapped = ema_to_slider_values(ema_df, ema_days)
            col_a, col_b = st.columns([1.05, 1])
            with col_a:
                st.subheader("생활패턴과 MDD 위험의 관련성")
                st.plotly_chart(plot_context_impact(ema_df), width="stretch")
                insights = lifestyle_insights(ema_df)
                if insights:
                    for item in insights:
                        klass = "risk" if item["delta"] > 0 else "protect"
                        sign = "+" if item["delta"] > 0 else "-"
                        st.markdown(
                            f"""
                            <div class="ema-insight {klass}">
                              <div class="title">{item['context']} · {item['direction']} ({sign}{abs(item['delta']):.1f}점)</div>
                              <div class="body">{item['summary']} 관련 회로: {item['circuit']} · 표본 {item['n']}회</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("생활패턴 체크가 충분하지 않아 관련성을 계산하기 어렵습니다.")
            with col_b:
                st.subheader("EMA가 움직인 MDD 회로")
                st.plotly_chart(plot_slider_estimates(mapped), width="stretch")
                st.caption("PFC 조절은 높을수록 건강 지표이며, 나머지는 높을수록 왜곡/부담 추정치입니다.")

            with st.expander("원자료 자세히 보기: 5개 EMA 문항과 시간대 패턴"):
                col_raw_a, col_raw_b = st.columns([1.1, 0.9])
                with col_raw_a:
                    st.plotly_chart(plot_ema_trajectory(ema_df), width="stretch")
                with col_raw_b:
                    st.plotly_chart(plot_window_patterns(ema_df), width="stretch")

            st.subheader("컨텍스트별 상세 수치")
            ctx = context_summary(ema_df)
            if ctx.empty:
                st.caption("체크된 컨텍스트가 충분하지 않습니다.")
            else:
                st.dataframe(
                    ctx.rename(
                        columns={
                            "context": "컨텍스트",
                            "n": "체크 수",
                            "risk_when_yes": "해당 시 위험 지수",
                            "risk_when_no": "비해당 시 위험 지수",
                            "delta": "차이",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    # ===== Tab 2: 삼각구조 (개요) =====
    with tab2:
        st.header("🔗 유전 → 뇌 → 행동 삼각구조")
        st.markdown("""
        **💡 읽는 법**: 위에서 아래로 읽으세요.
        - 🔴 **유전 변이**가 🧠 **뇌 회로**를 비정상으로 만들고
        - 🧠 비정상 뇌 회로가 😔 **우울 증상**을 만듭니다
        - 🔄 **점선** = 행동이 다시 유전 발현에 영향을 미치는 후성유전 피드백
        
        연결선 색: 🟢 정상 → 🟡 중간 → 🔴 비정상  |  뇌 노드 크기 = 왜곡도
        """)

        tri_fig, tri_dist = plot_simplified_triangle(slider_values, use_nonlinear)
        st.plotly_chart(tri_fig, width='stretch')

        st.subheader("🧠 핵심 회로별 왜곡도")
        short_names = {
            "dmn_overactivity": "DMN (반추)",
            "sgacc_hyperactivity": "sgACC (고착)",
            "reward_circuit_dysfunction": "보상 (무쾌감)",
            "amygdala_hypersensitivity": "편도체 (부정)",
            "salience_network_dysfunction": "SN (전환)",
            "pfc_dysregulation": "PFC (조절↓)",
            "hippocampus_atrophy": "해마 (기억↓)",
            "hpa_axis_dysregulation": "HPA (스트레스)",
        }
        cols4 = st.columns(4)
        for i, (ck, label) in enumerate(short_names.items()):
            d = tri_dist.get(ck, 0.5)
            icon = "🟢" if d < 0.35 else ("🟡" if d < 0.65 else "🔴")
            with cols4[i % 4]:
                st.metric(f"{icon} {label}", f"{d:.3f}")

    # ===== Tab 3: 뇌 회로 지도 =====
    with tab3:
        st.header("🧠 뇌 회로 지도 (해부학적 위치)")
        st.markdown("""
        반투명 뇌 외곽선 내부에 MDD 관련 뇌 영역을 실제 해부학적 위치에 배치했습니다.
        
        - 🔴 **빨간 선** = 과활성 연결  |  🔵 **파란 선** = 저활성  |  🟠 **주황 점선** = 조절장애
        - 노드 크기 = 왜곡도  |  **드래그**하여 회전, **스크롤**하여 확대/축소
        """)

        brain_fig, brain_dist = plot_brain_circuit(slider_values, use_nonlinear)
        st.plotly_chart(brain_fig, width='stretch')

        # 상세 네트워크 정보
        st.subheader("📊 네트워크별 상세")
        network_data = circuit_data.get("brain_networks", {})
        sel_net = st.selectbox(
            "네트워크 선택", list(network_data.keys()),
            format_func=lambda x: network_data[x].get("name", x),
        )
        if sel_net in network_data:
            net = network_data[sel_net]
            ca, cb = st.columns([1, 1])
            with ca:
                st.markdown(f"**📍 영역**: {', '.join(net['regions'])}")
                st.markdown(f"**✅ 정상**: {net['normal_function']}")
                st.markdown(f"**❌ MDD**: {net['mdd_distortion']}")
            with cb:
                st.markdown(f"**🏥 임상**: {net['clinical_implication']}")
                st.markdown(f"**📚 증거**: {net['evidence']}")
                st.markdown("**🎯 치료**:")
                for t in net.get("treatment_targets", []):
                    st.markdown(f"- {t}")

        # 상전이 곡선
        st.subheader("📊 상전이 곡선 (선형 vs 비선형)")
        st.markdown("""
        Yuan 2026 *Neuron*: 우울증은 **병리적 어트랙터** — 임계점을 넘으면
        빠져나오기 힘든 '우울 소용돌이'가 됩니다.
        """)
        sel_circuit = st.selectbox(
            "회로 선택", list(NAME_MAP.keys()),
            format_func=lambda x: NAME_MAP[x], key="phase_sel",
        )
        st.plotly_chart(plot_phase_transition(sel_circuit, ema_df), width='stretch')

    # ===== Tab 4: 다차원 원인 =====
    with tab4:
        st.header("🔍 다차원 원인 분석")

        ckeys_all = list(circuit_data["simulation_model"]["circuit_distortion_functions"].keys())
        all_dist = {k: calculate_circuit_distortion(slider_values, k, use_nonlinear)
                    for k in ckeys_all}
        all_pred = predict_symptoms(all_dist)

        for skey, sresult in all_pred.items():
            icon = "🔴" if sresult["severity"] == "높음" else "🟢"
            st.markdown(
                f"**{icon} {sresult['label']}**: "
                f"{sresult['severity']} (왜곡도: {sresult['distortion']:.3f})"
            )

        st.subheader("🧬 유전→뇌→행동 경로망")
        st.markdown("""
        | 유전 변이 | 측정 방식 | 주요 뇌 회로 영향 | 증거 강도 | 한계 |
        |-----------|----------|-------------------|----------|------|
        | **BDNF Met** | 🔴 범주형 (유전형) | 해마 위축, PFC 저하 | ★★★☆☆ | OR 1.2-1.5 (소효과) |
        | **FKBP5** | 🔴 범주형 (유전형) | HPA 축 항진, 염증↑ | ★★☆☆☆ | 주효과 OR=1.06, G×E에서만 의미 |
        | **5-HTTLPR** | 🔴 범주형 (유전형) | 편도체 과민 | ★☆☆☆☆ | 대규모 메타 재현 실패 |
        | **NR3C1** | 🟡 연속형 (메틸화%) | HPA 축 (GR↓) | ★★★☆☆ | 조직 특이성 |
        | **PRS** | 🟢 연속형 (백분위) | 전반적 취약성 | ★★★★★ | 설명 분산 ~8.9% |
        
        > 🔴 = 범주형 선택 · 🟡 = 연속형(특수검사) · 🟢 = 연속형(일반검사)
        """)

        st.subheader("🧠 Triple Network Model (Menon 2011)")
        st.markdown("""
        ```
        ┌──────────┐      ┌──────────┐
        │   DMN    │ ◀──▶ │  CEN/FPN │
        │ 반추 🔥   │      │ 인지통제 🎯│
        └────┬─────┘      └────┬─────┘
             │   SN 고장 ↓       │
             └───────┬──────────┘
              ┌──────▼──────┐
              │   SN 🚦     │
              │ 전환 스위치  │
              └─────────────┘

        MDD: SN 고장 → DMN에 갇힘 → 반추 지속
        ```
        """)

    # ===== Tab 5: 치료 가이드 =====
    with tab5:
        st.header("💊 치료/연구 가이드")

        tx_data = circuit_data.get("treatment_guide", {})
        for tx_name, tx in tx_data.items():
            with st.expander(f"**{tx_name}** — 증거: {tx['evidence_level']}"):
                st.markdown(f"**🎯 타겟 회로**: {', '.join(tx['targets'])}")
                st.markdown(f"**🔬 기전**: {tx['mechanism']}")
                st.markdown(f"**📝 비고**: {tx['notes']}")

        st.subheader("🧩 다노드 조합 치료 전략")
        st.markdown("""
        | 조합 | 타격 노드 | 효과 |
        |------|----------|------|
        | **SSRI + CBT** | 세로토닌↑ + DLPFC↑ | 편도체↓ + DMN↓ |
        | **SSRI + 운동** | 세로토닌↑ + BDNF↑ | 해마↑ + 보상↑ |
        | **rTMS + CBT** | DLPFC 자극 + 인지재구성 | SN↔CEN 복구 |
        | **케타민 + CBT** | 급속 가소성 + 인지 | 보상↑ + DMN↓ |

        > 💡 Yuan 2026: 치료는 **병리적 어트랙터에서 빠져나오는 힘**.
        > 다각 접근이 효과적인 이유는 여러 자기강화 루프를 동시에 끊기 때문.
        """)

    # ===== Tab 6: 사용 설명서 =====
    with tab6:
        st.header("📖 사용 설명서")
        st.markdown("""
        ## 🎯 이 도구의 목적

        MDD의 **유전-뇌-행동 삼각구조**를 시각화하여 복잡한 상호작용을 직관적으로 이해하기 위한
        **교육·연구 목적**의 도구입니다.

        ---

        ## 🧠 세 가지 시각화

        | 탭 | 내용 | 용도 |
        |----|------|------|
        | **삼각구조** | 유전→뇌→행동 핵심 경로 | 전체 구조 파악 |
        | **뇌 회로 지도** | 해부학적 위치의 뇌 영역 | 어떤 뇌 부위가 비정상인지 |
        | **증상 프로파일** | 레이더 차트 + DSM-5 | 슬라이더 조절 효과 확인 |

        ---

        ## 🧬 삼각구조란?

        ```
           [유전 취약성] ──→ [뇌 네트워크 이상] ──→ [행동/증상]
                ▲                                        │
                └──────── 후성유전 피드백 ◀───────────────┘
        ```

        Yuan et al. (2026) *Neuron*: 우울증은 **병리적 어트랙터**(빠져나오기 힘든 소용돌이).

        ---

        ## 🧠 뇌 회로 지도 읽는 법

        - 🔴 **빨간 선**: 과활성 연결 (예: DMN↔sgACC 과연결)
        - 🔵 **파란 선**: 저활성 연결 (예: DLPFC→편도체 억제↓)
        - 🟠 **주황 점선**: 조절 장애 (예: SN 전환 실패)
        - 노드 크기가 클수록 해당 영역의 왜곡도가 높음

        ---

        ## ⚠️ 주의사항

        1. **진단 도구가 아닙니다** — 교육·연구 목적
        2. **개인차가 큽니다** — MDD는 이질성이 높은 질환
        3. **슬라이더는 상대적 지표** — 실제 생물학적 측정값과 1:1 대응하지 않음
        4. **5-HTTLPR 논란** — 대규모 메타분석에서 재현 실패 (Risch 2009, JAMA)
        5. **FKBP5 제한** — 주효과 OR=1.062로 단독 예측력 거의 없음, G×E에서만 의미

        ---

        ## 🔴🟡🟢 신뢰도 라벨 안내

        각 지표 옆에 신뢰도 라벨이 표시됩니다:

        | 라벨 | 의미 | 예시 |
        |------|------|------|
        | 🟢 **높음** | 실제 검사로 측정, 확립된 컷오프 존재 | 코르티솔, CRP/IL-6, 해마체적, PRS |
        | 🟡 **중간** | fMRI/특수검사 필요, 노멋티브 데이터 제한적 | DMN, SN, sgACC, 보상회로, 편도체, 전두엽 |
        | 🔴 **낮음** | 범주형(유전형), 연속 척도 없음 | BDNF Met, FKBP5, 5-HTTLPR |

        > **유전 변이**(BDNF, FKBP5, 5-HTTLPR)는 **선택형(selectbox)** 으로 표시됩니다.
        > 유전형은 본질적으로 범주형(Val/Val, Val/Met, Met/Met 등)이므로 0-100 슬라이더가 과학적으로 부정확합니다.
        > 대신 **PRS(다유전 점수)** 는 정규분포를 따르므로 0-100 슬라이더로 표현 가능합니다.

        ---

        ## 📚 핵심 참고문헌

        1. Yuan et al. (2026). *Neuron* — 병리적 어트랙터 모델
        2. PGC MDD (2025). *Cell* — 697 loci, 685,808명 GWAS
        3. Menon (2011). *Brain* — Triple Network Model
        4. Zhang et al. (2026). *Psychol Med* — ALE 메타분석
        5. Border et al. (2019). *Am J Psychiatry* — 5-HTTLPR 재현 실패
        """)


if __name__ == "__main__":
    main()
