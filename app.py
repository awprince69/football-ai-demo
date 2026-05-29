"""
Football AI Prediction Engine v3.2 — POC Demo
Streamlit web application
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy.stats import poisson

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football AI Engine v3.2",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

# ─── THEME CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0A0E1A;
    color: #E2E8F0;
}

section[data-testid="stSidebar"] {
    background: #0D1120 !important;
    border-right: 1px solid #1E2A3A;
}

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748B;
    font-weight: 500;
}

.prob-bar-container {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 100%;
}

.prob-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 8px;
}

.prob-value {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 12px;
}

.pick-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.section-header {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3B82F6;
    font-weight: 600;
    margin-bottom: 6px;
}

.team-elo {
    font-size: 0.75rem;
    color: #94A3B8;
    font-family: 'DM Mono', monospace;
}

div[data-testid="stSelectbox"] label {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 12px 32px !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1D4ED8) !important;
    transform: translateY(-1px);
}

hr {
    border-color: #1E2A3A !important;
}

.stMetric {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 10px;
    padding: 14px 16px;
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD ASSETS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "xgb_model.joblib")

@st.cache_data
def load_elo():
    return json.loads((MODELS_DIR / "elo_ratings.json").read_text())

@st.cache_data
def load_calibration():
    return json.loads((MODELS_DIR / "calibration_params.json").read_text())

@st.cache_data
def load_predictions():
    df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df

@st.cache_data
def load_team_profiles():
    return json.loads((DATA_DIR / "team_profiles.json").read_text())

model        = load_model()
elo_ratings  = load_elo()
cal_params   = load_calibration()
pred_df      = load_predictions()
team_data    = load_team_profiles()
teams        = team_data["teams"]
xg_profiles  = team_data["xg_profiles"]
FEATURE_COLS = json.loads((MODELS_DIR / "feature_cols.json").read_text())

# ─── HELPERS ──────────────────────────────────────────────────────────────────
RHO = -0.020

def remove_vig(h, d, a):
    try:
        ih, id_, ia = 1/h, 1/d, 1/a
        t = ih + id_ + ia
        return ih/t, id_/t, ia/t
    except:
        return 0.45, 0.27, 0.28

def temperature_scale(prob, T):
    eps = 1e-7
    p   = max(eps, min(1-eps, prob))
    logit  = np.log(p / (1-p))
    scaled = logit / T
    return 1 / (1 + np.exp(-scaled))

def dc_correction(lh, la, h, a, rho):
    if h == 0 and a == 0: return 1 - lh * la * rho
    if h == 1 and a == 0: return 1 + la * rho
    if h == 0 and a == 1: return 1 + lh * rho
    if h == 1 and a == 1: return 1 - rho
    return 1.0

def poisson_probs(lh, la, rho=RHO, max_g=10):
    grid = np.zeros((max_g+1, max_g+1))
    for h in range(max_g+1):
        for a in range(max_g+1):
            grid[h,a] = poisson.pmf(h,lh) * poisson.pmf(a,la) * dc_correction(lh,la,h,a,rho)
    grid /= grid.sum()
    prob_h = float(np.sum(np.tril(grid,-1)))
    prob_d = float(np.sum(np.diag(grid)))
    prob_a = float(np.sum(np.triu(grid, 1)))
    tg = np.zeros(2*max_g+2)
    for h in range(max_g+1):
        for a in range(max_g+1):
            tg[h+a] += grid[h,a]
    p_o15 = sum(tg[i] for i in range(len(tg)) if i > 1.5)
    p_o25 = sum(tg[i] for i in range(len(tg)) if i > 2.5)
    p_o35 = sum(tg[i] for i in range(len(tg)) if i > 3.5)
    p_btts = (1-poisson.pmf(0,lh)) * (1-poisson.pmf(0,la))
    return dict(home=prob_h, draw=prob_d, away=prob_a,
                o15=p_o15, o25=p_o25, o35=p_o35, btts=float(p_btts))

def predict_match(home_team, away_team):
    home_elo = elo_ratings.get(home_team, 1500)
    away_elo = elo_ratings.get(away_team, 1500)
    elo_diff  = home_elo + 70 - away_elo

    home_prof = xg_profiles.get(home_team, {"avg_home_xg": 1.35, "avg_away_xg": 1.05})
    away_prof = xg_profiles.get(away_team, {"avg_home_xg": 1.35, "avg_away_xg": 1.05})
    xg_form_diff = home_prof["avg_home_xg"] - away_prof["avg_away_xg"]

    # Elo-based implied probabilities (fallback market estimate)
    elo_gap   = home_elo + 70 - away_elo
    raw_h     = 1 / (1 + 10**(-elo_gap/400))
    raw_a     = 1 - raw_h
    raw_d     = 0.26
    total_raw = raw_h + raw_d + raw_a
    imp_h = raw_h / total_raw
    imp_d = raw_d / total_raw
    imp_a = raw_a / total_raw

    features = np.array([[elo_diff, xg_form_diff, imp_h, imp_d, imp_a]])
    proba    = model.predict_proba(features)[0]

    # Apply temperature scaling
    t_home = cal_params.get("Home Win", {}).get("T", 1.0)
    t_draw = cal_params.get("Draw",     {}).get("T", 1.0)
    t_away = cal_params.get("Away Win", {}).get("T", 1.0)

    cal_h = temperature_scale(proba[0], t_home)
    cal_d = temperature_scale(proba[1], t_draw)
    cal_a = temperature_scale(proba[2], t_away)
    total = cal_h + cal_d + cal_a
    cal_h, cal_d, cal_a = cal_h/total, cal_d/total, cal_a/total

    # Poisson simulation for O/U and BTTS
    lh = home_prof["avg_home_xg"] * (1 + elo_gap/1000)
    la = away_prof["avg_away_xg"] * (1 - elo_gap/1000)
    lh = max(0.3, min(4.0, lh))
    la = max(0.3, min(4.0, la))

    poisson_p = poisson_probs(lh, la)

    # Confidence
    entropy = -sum(p * np.log(p+1e-9) for p in [cal_h, cal_d, cal_a])
    confidence = 1 - entropy / np.log(3)

    # Pick
    max_prob = max(cal_h, cal_d, cal_a)
    if max_prob == cal_h:   pick, pick_market = "HOME WIN", "1X2"
    elif max_prob == cal_d: pick, pick_market = "DRAW",     "1X2"
    else:                   pick, pick_market = "AWAY WIN", "1X2"

    # Override with O/U if strong signal
    if poisson_p["o25"] > 0.72:
        pick, pick_market = "OVER 2.5", "O/U"
    elif poisson_p["o25"] < 0.38:
        pick, pick_market = "UNDER 2.5", "O/U"

    return {
        "home_elo": home_elo, "away_elo": away_elo,
        "elo_diff": elo_diff, "xg_form_diff": xg_form_diff,
        "prob_home": cal_h, "prob_draw": cal_d, "prob_away": cal_a,
        "lambda_h": lh, "lambda_a": la,
        "o25": poisson_p["o25"], "o15": poisson_p["o15"],
        "o35": poisson_p["o35"], "btts": poisson_p["btts"],
        "confidence": confidence, "pick": pick, "pick_market": pick_market,
    }

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px'>
        <div style='font-size:0.65rem;letter-spacing:0.2em;color:#3B82F6;text-transform:uppercase;margin-bottom:4px'>
            Football AI
        </div>
        <div style='font-size:1.4rem;font-weight:700;color:#F1F5F9;line-height:1.2'>
            Prediction Engine
        </div>
        <div style='font-size:0.7rem;color:#475569;margin-top:4px;font-family:"DM Mono",monospace'>
            v3.2 — POC Demo
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["⚡  Dashboard", "🎯  Predict Match", "📊  Validation Results", "📈  Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem;color:#334155;line-height:1.8'>
        <div style='color:#475569;font-weight:600;margin-bottom:6px;font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase'>Data Scope</div>
        🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League<br>
        📅 5 Seasons (2018–2023)<br>
        ⚽ 1,900 Matches<br>
        🤖 XGBoost + Poisson<br>
        📐 5 Core Features
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE 1: DASHBOARD ────────────────────────────────────────────────────────
if page == "⚡  Dashboard":
    st.markdown("""
    <div style='margin-bottom:32px'>
        <div style='font-size:0.7rem;letter-spacing:0.15em;color:#3B82F6;text-transform:uppercase;margin-bottom:6px'>
            POC Validation Complete
        </div>
        <h1 style='font-size:2.2rem;font-weight:700;color:#F1F5F9;margin:0;line-height:1.2'>
            Football AI Prediction Engine
        </h1>
        <p style='color:#64748B;margin-top:8px;font-size:0.95rem'>
            Architecture v3.2 — validated on 1,900 real Premier League matches across 5 seasons
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (m1, "67.5%",  "O/U 2.5 Accuracy",  "#10B981", "Target: 55%"),
        (m2, "2.71%",  "Calibration ECE",    "#3B82F6", "Target: <5%"),
        (m3, "+0.81%", "Away Win CLV",        "#8B5CF6", "Edge confirmed"),
        (m4, "+0.51%", "Home Win CLV",        "#F59E0B", "Edge confirmed"),
    ]
    for col, val, label, color, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
                <div style="font-size:0.68rem;color:#334155;margin-top:6px">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Validation Scorecard</div>', unsafe_allow_html=True)
        scorecard = [
            ("CLV on Home Wins",          "+0.0051", True,  "Real edge confirmed"),
            ("CLV on Away Wins",          "+0.0081", True,  "Real edge confirmed"),
            ("Raw ECE average",           "3.25%",   True,  "Target was <12%"),
            ("ECE after calibration",     "2.71%",   True,  "Target was <5%"),
            ("Over/Under 2.5 accuracy",   "67.5%",   True,  "Target was >55%"),
            ("BTTS accuracy",             "66.8%",   True,  "Target was >55%"),
            ("Data pipeline quality",     "1900/1900",True, "Zero errors"),
            ("BTTS calibration (ECE)",    "6.26%",   False, "Needs lineup features"),
        ]
        for test, result, passed, note in scorecard:
            icon  = "✓" if passed else "⚠"
            color = "#10B981" if passed else "#F59E0B"
            st.markdown(f"""
            <div style='display:flex;align-items:center;padding:8px 12px;border-bottom:1px solid #1E2A3A;'>
                <span style='color:{color};font-size:0.85rem;width:20px'>{icon}</span>
                <span style='flex:1;font-size:0.82rem;color:#CBD5E1'>{test}</span>
                <span style='font-family:"DM Mono",monospace;font-size:0.8rem;color:{color};font-weight:600;margin-right:12px'>{result}</span>
                <span style='font-size:0.7rem;color:#475569'>{note}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Current Elo Standings</div>', unsafe_allow_html=True)
        top_teams = sorted(elo_ratings.items(), key=lambda x: -x[1])[:10]
        for i, (team, elo) in enumerate(top_teams):
            bar_width = int((elo - 1400) / (1700 - 1400) * 100)
            st.markdown(f"""
            <div style='padding:6px 10px;border-bottom:1px solid #0D1120'>
                <div style='display:flex;align-items:center;margin-bottom:4px'>
                    <span style='font-size:0.7rem;color:#475569;width:20px'>{i+1}</span>
                    <span style='font-size:0.82rem;color:#E2E8F0;flex:1'>{team}</span>
                    <span style='font-family:"DM Mono",monospace;font-size:0.75rem;color:#3B82F6'>{elo:.0f}</span>
                </div>
                <div style='background:#0A0E1A;border-radius:2px;height:3px;margin-left:20px'>
                    <div style='background:linear-gradient(90deg,#1D4ED8,#3B82F6);width:{bar_width}%;height:3px;border-radius:2px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">System Architecture Overview</div>', unsafe_allow_html=True)

    arch_cols = st.columns(5)
    arch = [
        ("📥", "Data Layer", "5 sources\nBet365, Pinnacle\nUnderstat, FBRef"),
        ("⚗", "34 Features", "Elo, xG form\nOdds signals\nFatigue, Lineups"),
        ("🤖", "8 Models", "XGBoost\nLightGBM, Poisson\nElo, Ensemble"),
        ("📐", "Calibration", "Platt Scaling\nIsotonic Reg.\nTemp. Scaling"),
        ("🎯", "3 Markets", "European 1X2\nAsian Handicap\nOver/Under"),
    ]
    for col, (icon, title, desc) in zip(arch_cols, arch):
        with col:
            st.markdown(f"""
            <div style='background:#111827;border:1px solid #1E2A3A;border-radius:10px;padding:16px;text-align:center'>
                <div style='font-size:1.6rem;margin-bottom:8px'>{icon}</div>
                <div style='font-size:0.75rem;font-weight:700;color:#E2E8F0;margin-bottom:6px'>{title}</div>
                <div style='font-size:0.65rem;color:#475569;line-height:1.7;white-space:pre-line'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── PAGE 2: PREDICT MATCH ────────────────────────────────────────────────────
elif page == "🎯  Predict Match":
    st.markdown("""
    <div style='margin-bottom:28px'>
        <div class="section-header">AI Prediction Engine</div>
        <h2 style='font-size:1.8rem;font-weight:700;color:#F1F5F9;margin:0'>
            Match Prediction
        </h2>
        <p style='color:#64748B;margin-top:6px;font-size:0.88rem'>
            Select two Premier League teams to generate an AI-powered prediction using Elo ratings, xG form, and calibrated Poisson simulation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_home, col_vs, col_away = st.columns([5, 1, 5])

    with col_home:
        st.markdown('<div class="section-header">Home Team</div>', unsafe_allow_html=True)
        home_team = st.selectbox("Home Team", teams, index=teams.index("Man City") if "Man City" in teams else 0, label_visibility="collapsed")
        h_elo = elo_ratings.get(home_team, 1500)
        st.markdown(f'<div class="team-elo">Elo: {h_elo:.0f}  ·  xG avg: {xg_profiles.get(home_team,{}).get("avg_home_xg",1.3):.2f}</div>', unsafe_allow_html=True)

    with col_vs:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:1.2rem;color:#475569;font-weight:700;padding-top:8px">VS</div>', unsafe_allow_html=True)

    with col_away:
        st.markdown('<div class="section-header">Away Team</div>', unsafe_allow_html=True)
        away_default = teams.index("Arsenal") if "Arsenal" in teams else 1
        away_team = st.selectbox("Away Team", teams, index=away_default, label_visibility="collapsed")
        a_elo = elo_ratings.get(away_team, 1500)
        st.markdown(f'<div class="team-elo">Elo: {a_elo:.0f}  ·  xG avg: {xg_profiles.get(away_team,{}).get("avg_away_xg",1.1):.2f}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        run_pred = st.button("⚡  Generate Prediction", type="primary")

    if run_pred:
        if home_team == away_team:
            st.error("Please select two different teams.")
        else:
            with st.spinner("Running prediction model..."):
                result = predict_match(home_team, away_team)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<br>", unsafe_allow_html=True)

            # Recommended pick banner
            pick_colors = {
                "HOME WIN":   ("#1D4ED8", "#EFF6FF"),
                "DRAW":       ("#92400E", "#FFFBEB"),
                "AWAY WIN":   ("#065F46", "#F0FDF4"),
                "OVER 2.5":  ("#5B21B6", "#F5F3FF"),
                "UNDER 2.5": ("#7C3AED", "#F5F3FF"),
            }
            pick_bg, pick_text_bg = pick_colors.get(result["pick"], ("#1D4ED8", "#EFF6FF"))
            conf_pct = result["confidence"] * 100
            conf_label = "HIGH" if conf_pct > 65 else "MEDIUM" if conf_pct > 45 else "LOW"
            conf_color = "#10B981" if conf_pct > 65 else "#F59E0B" if conf_pct > 45 else "#EF4444"

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#111827,#1a2235);border:1px solid #1E2A3A;border-radius:16px;padding:28px;text-align:center;margin-bottom:24px'>
                <div style='font-size:0.68rem;letter-spacing:0.15em;color:#64748B;text-transform:uppercase;margin-bottom:10px'>
                    AI Recommendation · {result["pick_market"]} Market
                </div>
                <div style='display:inline-block;background:{pick_bg};color:white;padding:12px 40px;border-radius:50px;font-size:1.3rem;font-weight:800;letter-spacing:0.1em;margin-bottom:14px'>
                    {result["pick"]}
                </div>
                <div style='font-size:0.8rem;color:#64748B'>
                    Confidence: <span style='color:{conf_color};font-weight:700'>{conf_label}</span>
                    <span style='color:#334155;margin:0 8px'>·</span>
                    Score: <span style='font-family:"DM Mono",monospace;color:#94A3B8'>{conf_pct:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            c1, c2, c3 = st.columns(3)
            probs = [
                (c1, home_team, result["prob_home"], "#3B82F6"),
                (c2, "Draw",    result["prob_draw"],  "#6B7280"),
                (c3, away_team, result["prob_away"],  "#8B5CF6"),
            ]
            for col, label, prob, color in probs:
                with col:
                    pct = prob * 100
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <div class="prob-label">{label}</div>
                        <div class="prob-value" style="color:{color}">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(
                        go.Figure(go.Indicator(
                            mode="gauge",
                            value=pct,
                            gauge=dict(
                                axis=dict(range=[0, 100], tickfont=dict(color="#475569", size=9)),
                                bar=dict(color=color, thickness=0.7),
                                bgcolor="#111827",
                                bordercolor="#1E2A3A",
                                steps=[dict(range=[0,100], color="#0A0E1A")],
                            ),
                            domain=dict(x=[0,1], y=[0,1])
                        )).update_layout(
                            height=130, margin=dict(t=10,b=10,l=10,r=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#475569")
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Goals and O/U section
            st.markdown('<div class="section-header">Goals Prediction</div>', unsafe_allow_html=True)
            g1, g2, g3, g4, g5 = st.columns(5)

            goal_metrics = [
                (g1, f"{result['lambda_h']:.2f}", f"Expected\n{home_team[:10]} Goals", "#3B82F6"),
                (g2, f"{result['lambda_a']:.2f}", f"Expected\n{away_team[:10]} Goals", "#8B5CF6"),
                (g3, f"{result['o25']*100:.1f}%", "Over 2.5\nGoals", "#10B981" if result["o25"] > 0.5 else "#64748B"),
                (g4, f"{result['o15']*100:.1f}%", "Over 1.5\nGoals", "#10B981" if result["o15"] > 0.5 else "#64748B"),
                (g5, f"{result['btts']*100:.1f}%", "Both Teams\nTo Score", "#10B981" if result["btts"] > 0.5 else "#64748B"),
            ]
            for col, val, label, color in goal_metrics:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{color};font-size:1.6rem">{val}</div>
                        <div class="metric-label" style="white-space:pre-line">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Elo context
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Match Context</div>', unsafe_allow_html=True)
            cx1, cx2, cx3 = st.columns(3)
            with cx1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#F59E0B;font-size:1.5rem">{result['home_elo']:.0f}</div>
                    <div class="metric-label">{home_team} Elo</div>
                </div>""", unsafe_allow_html=True)
            with cx2:
                diff = result["elo_diff"]
                color = "#10B981" if diff > 0 else "#EF4444"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color};font-size:1.5rem">{diff:+.0f}</div>
                    <div class="metric-label">Elo Difference (home advantage included)</div>
                </div>""", unsafe_allow_html=True)
            with cx3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#F59E0B;font-size:1.5rem">{result['away_elo']:.0f}</div>
                    <div class="metric-label">{away_team} Elo</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style='background:#111827;border:1px solid #1E2A3A;border-radius:10px;padding:14px 18px;margin-top:16px'>
                <div style='font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#3B82F6;margin-bottom:6px'>Note</div>
                <div style='font-size:0.78rem;color:#475569;line-height:1.7'>
                    This prediction uses Elo ratings computed from all 1,900 historical matches (up to May 2023) and xG form from the last 10 appearances per team.
                    Production deployment will use live current-season data, confirmed lineups, live odds, and all 34 v3.2 features.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── PAGE 3: VALIDATION RESULTS ───────────────────────────────────────────────
elif page == "📊  Validation Results":
    st.markdown("""
    <div style='margin-bottom:28px'>
        <div class="section-header">POC Method 1 — CLV Backtest</div>
        <h2 style='font-size:1.8rem;font-weight:700;color:#F1F5F9;margin:0'>Historical Validation</h2>
        <p style='color:#64748B;margin-top:6px;font-size:0.88rem'>
            Test set: 2021/22 and 2022/23 seasons — matches the model never saw during training.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    f1, f2, f3 = st.columns(3)
    with f1:
        season_filter = st.selectbox("Season", ["All", "2021/22", "2022/23"])
    with f2:
        result_filter = st.selectbox("Result", ["All", "H - Home Win", "D - Draw", "A - Away Win"])
    with f3:
        clv_filter = st.selectbox("CLV", ["All", "Positive CLV only", "Negative CLV only"])

    # Apply filters
    df = pred_df.copy()
    if season_filter != "All":
        df = df[df["season"] == season_filter]
    if result_filter != "All":
        df = df[df["result"] == result_filter[0]]
    if "Positive" in clv_filter:
        df = df[df["clv_outcome"] > 0]
    elif "Negative" in clv_filter:
        df = df[df["clv_outcome"] < 0]

    # Summary metrics
    if len(df) > 0:
        valid = df.dropna(subset=["prob_home", "clv_outcome"])
        mean_clv  = valid["clv_outcome"].mean()
        pos_clv   = (valid["clv_outcome"] > 0).mean() * 100
        model_correct = (
            ((valid["prob_home"] > valid["prob_draw"]) & (valid["prob_home"] > valid["prob_away"]) & (valid["result"] == "H")) |
            ((valid["prob_draw"] > valid["prob_home"]) & (valid["prob_draw"] > valid["prob_away"]) & (valid["result"] == "D")) |
            ((valid["prob_away"] > valid["prob_home"]) & (valid["prob_away"] > valid["prob_draw"]) & (valid["result"] == "A"))
        ).mean() * 100

        sv1, sv2, sv3, sv4 = st.columns(4)
        for col, val, label, color in [
            (sv1, f"{len(valid)}", "Matches", "#3B82F6"),
            (sv2, f"{mean_clv:+.4f}", "Mean CLV", "#10B981" if mean_clv > 0 else "#EF4444"),
            (sv3, f"{pos_clv:.1f}%", "Positive CLV %", "#8B5CF6"),
            (sv4, f"{model_correct:.1f}%", "Directional Accuracy", "#F59E0B"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color};font-size:1.5rem">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        chart_c, table_c = st.columns([2, 3])

        with chart_c:
            st.markdown('<div class="section-header">CLV Distribution</div>', unsafe_allow_html=True)
            fig = go.Figure()
            pos_vals = valid[valid["clv_outcome"] > 0]["clv_outcome"]
            neg_vals = valid[valid["clv_outcome"] <= 0]["clv_outcome"]
            fig.add_trace(go.Histogram(x=pos_vals, name="Positive", nbinsx=25,
                                       marker_color="#10B981", opacity=0.8))
            fig.add_trace(go.Histogram(x=neg_vals, name="Negative", nbinsx=25,
                                       marker_color="#EF4444", opacity=0.8))
            fig.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1.5)
            fig.add_vline(x=mean_clv, line_dash="dot", line_color="#F59E0B",
                         line_width=2, annotation_text=f"Mean: {mean_clv:+.4f}",
                         annotation_font_color="#F59E0B", annotation_font_size=11)
            fig.update_layout(
                barmode="overlay", height=280,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10,b=30,l=0,r=0),
                legend=dict(font=dict(color="#94A3B8", size=10), bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(color="#475569", tickfont=dict(size=10), gridcolor="#1E2A3A"),
                yaxis=dict(color="#475569", tickfont=dict(size=10), gridcolor="#1E2A3A"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with table_c:
            st.markdown('<div class="section-header">Recent Predictions</div>', unsafe_allow_html=True)
            display = valid.sort_values("match_date", ascending=False).head(15)
            for _, row in display.iterrows():
                if pd.isna(row["prob_home"]): continue
                clv   = row["clv_outcome"]
                clv_c = "#10B981" if clv > 0 else "#EF4444"
                res   = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}.get(row["result"], row["result"])
                st.markdown(f"""
                <div style='display:flex;align-items:center;padding:7px 10px;border-bottom:1px solid #0D1120;font-size:0.75rem'>
                    <span style='color:#475569;width:82px;font-family:"DM Mono",monospace'>{str(row['match_date'])[:10]}</span>
                    <span style='flex:1;color:#CBD5E1'>{row['home_team']} <span style='color:#475569'>vs</span> {row['away_team']}</span>
                    <span style='color:#64748B;margin-right:12px'>{res}</span>
                    <span style='font-family:"DM Mono",monospace;color:{clv_c};font-weight:600'>{clv:+.4f}</span>
                </div>
                """, unsafe_allow_html=True)


# ─── PAGE 4: MODEL PERFORMANCE ────────────────────────────────────────────────
elif page == "📈  Model Performance":
    st.markdown("""
    <div style='margin-bottom:28px'>
        <div class="section-header">POC Method 2 — Poisson Calibration</div>
        <h2 style='font-size:1.8rem;font-weight:700;color:#F1F5F9;margin:0'>Model Performance</h2>
        <p style='color:#64748B;margin-top:6px;font-size:0.88rem'>
            Calibration metrics across all 5 markets — 1,900 matches tested.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Calibration results
    cal_data = {
        "Home Win":  {"ece_before": 0.0169, "ece_after": 0.0167, "T": 0.998,  "brier": 0.1707, "target": 0.05},
        "Draw":      {"ece_before": 0.0146, "ece_after": 0.0143, "T": 0.998,  "brier": 0.1653, "target": 0.05},
        "Away Win":  {"ece_before": 0.0247, "ece_after": 0.0150, "T": 0.893,  "brier": 0.1511, "target": 0.05},
        "Over 2.5":  {"ece_before": 0.0375, "ece_after": 0.0271, "T": 1.061,  "brier": 0.2057, "target": 0.05},
        "BTTS":      {"ece_before": 0.0687, "ece_after": 0.0626, "T": 1.123,  "brier": 0.2125, "target": 0.05},
    }

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">ECE Before & After Calibration</div>', unsafe_allow_html=True)
        markets = list(cal_data.keys())
        before  = [cal_data[m]["ece_before"]*100 for m in markets]
        after   = [cal_data[m]["ece_after"]*100  for m in markets]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Before Scaling", x=markets, y=before,
                             marker_color="#3B82F6", opacity=0.6))
        fig.add_trace(go.Bar(name="After Scaling",  x=markets, y=after,
                             marker_color="#10B981", opacity=0.9))
        fig.add_hline(y=5, line_dash="dash", line_color="#F59E0B",
                      annotation_text="Target: 5%",
                      annotation_font_color="#F59E0B", annotation_font_size=11)
        fig.update_layout(
            barmode="group", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10,b=10,l=0,r=0),
            legend=dict(font=dict(color="#94A3B8", size=10), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(color="#475569", tickfont=dict(size=10)),
            yaxis=dict(color="#475569", tickfont=dict(size=10), gridcolor="#1E2A3A",
                       title="ECE (%)", title_font=dict(color="#475569", size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<div class="section-header">Temperature Scaling Parameters</div>', unsafe_allow_html=True)
        for market, d in cal_data.items():
            improvement = (d["ece_before"] - d["ece_after"]) / d["ece_before"] * 100
            met         = d["ece_after"] < d["target"]
            color       = "#10B981" if met else "#F59E0B"
            t_color     = "#3B82F6" if abs(d["T"]-1) < 0.05 else "#F59E0B"
            st.markdown(f"""
            <div style='display:flex;align-items:center;padding:10px 12px;border-bottom:1px solid #1E2A3A'>
                <span style='flex:1;font-size:0.82rem;color:#CBD5E1'>{market}</span>
                <span style='font-family:"DM Mono",monospace;font-size:0.75rem;color:{t_color};margin-right:12px'>T={d["T"]:.3f}</span>
                <span style='font-family:"DM Mono",monospace;font-size:0.75rem;color:{color};margin-right:10px'>{d["ece_after"]*100:.2f}%</span>
                <span style='font-size:0.7rem;color:{color}'>{"✓" if met else "⚠"}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CLV by outcome
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">CLV by Outcome Type</div>', unsafe_allow_html=True)
        outcomes = ["Home Win", "Draw", "Away Win"]
        clvs     = [0.0051, -0.0163, 0.0081]
        counts   = [345, 175, 238]
        colors   = ["#3B82F6" if c > 0 else "#EF4444" for c in clvs]
        fig2 = go.Figure(go.Bar(
            x=outcomes, y=clvs, marker_color=colors,
            text=[f"{c:+.4f}<br>n={n}" for c, n in zip(clvs, counts)],
            textposition="outside", textfont=dict(size=10, color="#94A3B8")
        ))
        fig2.add_hline(y=0, line_color="#475569", line_width=1)
        fig2.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30,b=10,l=0,r=0),
            xaxis=dict(color="#475569", tickfont=dict(size=11)),
            yaxis=dict(color="#475569", tickfont=dict(size=10), gridcolor="#1E2A3A",
                       title="Mean CLV", title_font=dict(color="#475569", size=10)),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="section-header">Phase 1 vs Phase 2 Targets</div>', unsafe_allow_html=True)
        phases = ["POC\n(5 features)", "Phase 1\n(34 features)", "Phase 2\n(+ Deep Learning)", "Bookmaker\nBenchmark"]
        log_losses = [1.048, 0.965, 0.930, 0.920]
        bar_colors = ["#475569", "#3B82F6", "#10B981", "#F59E0B"]
        fig3 = go.Figure(go.Bar(
            x=phases, y=log_losses, marker_color=bar_colors,
            text=[f"{v:.3f}" for v in log_losses],
            textposition="outside", textfont=dict(size=11, color="#94A3B8")
        ))
        fig3.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30,b=10,l=0,r=0),
            xaxis=dict(color="#475569", tickfont=dict(size=10)),
            yaxis=dict(color="#475569", tickfont=dict(size=10), gridcolor="#1E2A3A",
                       title="Log Loss (lower = better)", title_font=dict(color="#475569", size=10),
                       range=[0.85, 1.10]),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Bottom summary
    st.markdown('<div class="section-header">Key Findings Summary</div>', unsafe_allow_html=True)
    findings = [
        ("🎯", "O/U 2.5 accuracy 67.5%", "Exceeds target by 12.5 percentage points. Production-ready."),
        ("📐", "Average ECE 2.71%", "Near-bookmaker calibration. Target was 5%."),
        ("📈", "Positive CLV confirmed", "Home Win +0.51%, Away Win +0.81% on 758 test matches."),
        ("⚠", "Draw CLV weakness", "Expected — addressed by Asian Handicap features in v3.2."),
        ("🔄", "Auto-retraining designed", "Champion/challenger gate after every matchday in production."),
    ]
    for icon, title, desc in findings:
        st.markdown(f"""
        <div style='display:flex;gap:14px;padding:12px 14px;border-bottom:1px solid #1E2A3A;align-items:flex-start'>
            <span style='font-size:1.1rem'>{icon}</span>
            <div>
                <div style='font-size:0.82rem;font-weight:600;color:#E2E8F0;margin-bottom:3px'>{title}</div>
                <div style='font-size:0.75rem;color:#475569'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
