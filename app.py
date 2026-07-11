"""Four Factors prediction dashboard (project demo artifact).

Demonstrates the deployment logistic-regression model on two held-out seasons. Games for a
chosen date are shown as match cards (team logos + each team's predicted win %); selecting a
card shows the factor breakdown behind the prediction and lets the actual result be revealed.
Plus a per-season accuracy tally. This is a demonstration of the pipeline, not a claim of
predictive strength.

Run from the project root:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

SEASONS = ["2024-25", "2025-26"]
FACTOR_LABELS = {
    "contrib_eFG": "eFG%",
    "contrib_TOV_pct": "TOV%",
    "contrib_ORB_pct": "ORB%",
    "contrib_FTrate": "FTr",
}
HOME_COLOR = "#1f77b4"
AWAY_COLOR = "#d62728"

# NBA abbreviation -> ESPN logo slug (all 30 verified).
LOGO_SLUG = {
    "ATL": "atl", "BKN": "bkn", "BOS": "bos", "CHA": "cha", "CHI": "chi", "CLE": "cle",
    "DAL": "dal", "DEN": "den", "DET": "det", "GSW": "gs", "HOU": "hou", "IND": "ind",
    "LAC": "lac", "LAL": "lal", "MEM": "mem", "MIA": "mia", "MIL": "mil", "MIN": "min",
    "NOP": "no", "NYK": "ny", "OKC": "okc", "ORL": "orl", "PHI": "phi", "PHX": "phx",
    "POR": "por", "SAC": "sac", "SAS": "sa", "TOR": "tor", "UTA": "utah", "WAS": "wsh",
}


def logo(abbr):
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{LOGO_SLUG[abbr]}.png"


@st.cache_data
def load_predictions(season):
    df = pd.read_csv(PROCESSED_DIR / f"05_predictions_{season}.csv", parse_dates=["Date"])
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_tally():
    return pd.read_csv(PROCESSED_DIR / "05_tally.csv")


st.set_page_config(page_title="Four Factors Prediction Demo", layout="centered")
st.title("Four Factors — NBA Game Prediction Demo")
st.caption(
    "Demonstration of the Four Factors logistic-regression model on held-out seasons. "
    "Each season is predicted by a model trained only on earlier seasons (walk-forward, "
    "leakage-free). This is a demo of the pipeline, **not** a claim of predictive strength."
)

# --- Season + date selectors ---
top = st.columns(2)
season = top[0].selectbox("Season", SEASONS)
preds = load_predictions(season)

dates = sorted(preds["Date"].dt.date.unique())
date_labels = {d.strftime("%a, %b %d %Y"): d for d in dates}
chosen_label = top[1].selectbox("Date", list(date_labels))
chosen_date = date_labels[chosen_label]

games_today = preds[preds["Date"].dt.date == chosen_date].reset_index(drop=True)
gids = list(games_today["GAME_ID"])

# Keep a valid selection for the current view.
if st.session_state.get("sel_gid") not in gids:
    st.session_state["sel_gid"] = gids[0]

# --- Match cards (win % on each side instead of tip-off time) ---
st.subheader(f"Games — {chosen_label}")
for _, g in games_today.iterrows():
    gid = g["GAME_ID"]
    home, away = g["home_team"], g["away_team"]
    home_pct = g["p_home_win"] * 100
    away_pct = 100 - home_pct
    selected = gid == st.session_state["sel_gid"]

    with st.container(border=True):
        c = st.columns([1, 2.4, 0.6, 2.4, 1, 1.6], vertical_alignment="center")
        c[0].image(logo(away), width=44)
        c[1].markdown(
            f"**{away}**<br><span style='color:{AWAY_COLOR}'>{away_pct:.0f}% win</span>",
            unsafe_allow_html=True,
        )
        c[2].markdown("<div style='text-align:center'>@</div>", unsafe_allow_html=True)
        c[3].markdown(
            f"<div style='text-align:right'><b>{home}</b><br>"
            f"<span style='color:{HOME_COLOR}'>{home_pct:.0f}% win</span></div>",
            unsafe_allow_html=True,
        )
        c[4].image(logo(home), width=44)
        label = "● Selected" if selected else "View ▸"
        if c[5].button(label, key=f"btn_{gid}", type="primary" if selected else "secondary",
                       use_container_width=True):
            st.session_state["sel_gid"] = gid
            st.rerun()

# --- Detail panel for the selected game ---
game = games_today[games_today["GAME_ID"] == st.session_state["sel_gid"]].iloc[0]
home, away = game["home_team"], game["away_team"]
p_home = float(game["p_home_win"])

st.divider()
st.subheader(f"Prediction — {away} @ {home}")
favored = home if p_home >= 0.5 else away
fav_prob = p_home if p_home >= 0.5 else 1 - p_home
st.metric(f"Home win probability — {home}", f"{p_home * 100:.1f}%")
st.progress(p_home)
st.write(f"Model leans toward **{favored}** ({fav_prob * 100:.1f}%).")

# Factor breakdown (interpretability centrepiece).
st.markdown("**What drove the prediction**")
st.caption(
    "Each bar is a factor's contribution (standardized coefficient x this game's standardized "
    f"differential). Right favours **{home}**, left favours **{away}**."
)
contribs = [(FACTOR_LABELS[col], float(game[col])) for col in FACTOR_LABELS]
labels = [c[0] for c in contribs]
values = [c[1] for c in contribs]
colors = [HOME_COLOR if v >= 0 else AWAY_COLOR for v in values]

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.barh(labels, values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(f"<- favours {away}      favours {home} ->")
ax.invert_yaxis()
fig.tight_layout()
st.pyplot(fig)

# Reveal.
if st.button("Reveal actual result"):
    actual_winner = home if game["actual_home_win"] == 1 else away
    st.write(f"Final result: **{actual_winner} won** "
             f"({home} {'won' if game['actual_home_win'] == 1 else 'lost'} at home).")
    if game["correct"] == 1:
        st.success("Prediction was CORRECT.")
    else:
        st.error("Prediction was incorrect.")
else:
    st.info("Result hidden — reveal it after reading the prediction.")

# --- Season tally ---
st.divider()
st.subheader(f"Season tally — {season}")
tally = load_tally()
row = tally[tally["season"] == season].iloc[0]
t1, t2 = st.columns(2)
t1.metric(
    "Model accuracy",
    f"{row['model_accuracy'] * 100:.1f}%",
    help=f"{int(row['model_correct'])} of {int(row['n_games'])} games predicted correctly",
)
t2.metric("Always-pick-home baseline", f"{row['always_home_accuracy'] * 100:.1f}%")
st.caption(
    f"Model predicted {int(row['model_correct'])} of {int(row['n_games'])} games correctly "
    f"({row['model_accuracy'] * 100:.1f}%), vs {row['always_home_accuracy'] * 100:.1f}% for "
    "always picking the home team."
)
