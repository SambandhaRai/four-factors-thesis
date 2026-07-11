"""Four Factors prediction dashboard (project demo artifact).

Demonstrates the deployment logistic-regression model on two held-out seasons, organised
in two tabs. 'Game predictions' shows the games for a chosen date as match cards (team
logos + each team's predicted win %); selecting a card shows the factor breakdown behind
the prediction and lets the actual result be revealed. 'Season performance' shows the
accuracy tally, cumulative accuracy over the season, and accuracy by model confidence.
This is a demonstration of the pipeline, not a claim of predictive strength.

Run from the project root:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
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
FORM_COLS = {"eFG%": "eFG", "TOV%": "TOV_pct", "ORB%": "ORB_pct", "FTr": "FTrate"}
FRIENDLY_NAMES = {
    "eFG%": "Shooting (eFG%)",
    "TOV%": "Turnovers (TOV%)",
    "ORB%": "Off. rebounds (ORB%)",
    "FTr": "Free throws (FTr)",
    "Home court": "Home court",
}
HOME_COLOR = "#1f77b4"
AWAY_COLOR = "#d62728"
NEUTRAL_COLOR = "#7f7f7f"

# NBA abbreviation -> full team name.
TEAM_NAME = {
    "ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

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
    # GAME_ID is read as a string to keep leading zeros (needed for the form-table join).
    df = pd.read_csv(
        PROCESSED_DIR / f"05_predictions_{season}.csv",
        parse_dates=["Date"], dtype={"GAME_ID": str},
    )
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_form():
    # Rolling-average factor values each team carried into the game (leakage-free features).
    cols = ["GAME_ID"] + [f"{side}_{suffix}" for suffix in FORM_COLS.values()
                          for side in ("home", "away")]
    df = pd.read_csv(PROCESSED_DIR / "team_games_features_deploy.csv",
                     usecols=cols, dtype={"GAME_ID": str})
    return df.set_index("GAME_ID")


@st.cache_data
def load_tally():
    return pd.read_csv(PROCESSED_DIR / "05_tally.csv")


st.set_page_config(page_title="Four Factors Prediction Demo", layout="wide")
st.title("Four Factors — NBA Game Prediction Demo")
st.caption(
    "Demonstration of the Four Factors logistic-regression model on held-out seasons. "
    "Each season is predicted by a model trained only on earlier seasons (walk-forward, "
    "leakage-free). This is a demo of the pipeline, **not** a claim of predictive strength."
)

# Season selector applies to both tabs.
season = st.columns(3)[0].selectbox("Season", SEASONS)
preds = load_predictions(season)

tab_games, tab_perf = st.tabs(["Game predictions", "Season performance"])

# =====================================================================
# Tab 1 — Game predictions
# =====================================================================
with tab_games:
    dates = sorted(preds["Date"].dt.date.unique())
    date_labels = {d.strftime("%a, %b %d %Y"): d for d in dates}
    chosen_label = st.columns(3)[0].selectbox("Date", list(date_labels))
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
        home_name, away_name = TEAM_NAME[home], TEAM_NAME[away]
        home_pct = g["p_home_win"] * 100
        away_pct = 100 - home_pct
        selected = gid == st.session_state["sel_gid"]

        with st.container(border=True):
            c = st.columns([1, 2.4, 0.6, 2.4, 1, 1.6], vertical_alignment="center")
            c[0].image(logo(home), width=44)
            c[1].markdown(
                f"<div style='text-align:center'><b>{home_name}</b><br>"
                f"<span style='color:{HOME_COLOR}'>{home_pct:.0f}% win</span></div>",
                unsafe_allow_html=True,
            )
            c[2].markdown(
                "<div style='text-align:center;font-size:20px;font-weight:700'>vs</div>",
                unsafe_allow_html=True,
            )
            c[3].markdown(
                f"<div style='text-align:center'><b>{away_name}</b><br>"
                f"<span style='color:{AWAY_COLOR}'>{away_pct:.0f}% win</span></div>",
                unsafe_allow_html=True,
            )
            c[4].image(logo(away), width=44)
            label = "● Selected" if selected else "View ▸"
            if c[5].button(label, key=f"btn_{gid}", type="primary" if selected else "secondary",
                           width="stretch"):
                st.session_state["sel_gid"] = gid
                st.rerun()

    # --- Detail panel for the selected game ---
    game = games_today[games_today["GAME_ID"] == st.session_state["sel_gid"]].iloc[0]
    home, away = game["home_team"], game["away_team"]
    home_name, away_name = TEAM_NAME[home], TEAM_NAME[away]
    p_home = float(game["p_home_win"])
    p_away = 1 - p_home

    st.divider()
    st.subheader(f"Prediction — {home_name} vs {away_name}")
    favored = home_name if p_home >= 0.5 else away_name
    fav_prob = max(p_home, p_away)

    # Two-sided probability bar: home share on the left, away share on the right.
    st.markdown(
        f"""
        <div style='display:flex;height:30px;border-radius:6px;overflow:hidden;
                    font-size:14px;font-weight:600;color:white;margin-bottom:6px'>
          <div style='width:{p_home * 100:.1f}%;background:{HOME_COLOR};
                      padding:5px 10px;white-space:nowrap'>{home_name} {p_home * 100:.0f}%</div>
          <div style='width:{p_away * 100:.1f}%;background:{AWAY_COLOR};padding:5px 10px;
                      text-align:right;white-space:nowrap'>{away_name} {p_away * 100:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(f"Model leans toward **{favored}** ({fav_prob * 100:.1f}%).")

    # Factor breakdown (interpretability centrepiece).
    st.markdown("**What drove the prediction**")
    st.caption(
        f"Each factor pulls the prediction toward **{home_name}** (left, blue) or **{away_name}** "
        "(right, red). The percentage is that factor's share of the model's total reasoning "
        "for this game — the five shares add up to 100%, separate from the win probabilities "
        "above. 'Home court' is the model's built-in edge for the home team."
    )
    contribs = [(FACTOR_LABELS[col], float(game[col])) for col in FACTOR_LABELS]

    # The intercept (home-court baseline) is recovered from the final log-odds minus the
    # four factor contributions, so it reflects whichever season model made the prediction.
    logit = np.log(p_home / p_away)
    home_court = logit - sum(v for _, v in contribs)
    contribs.append(("Home court", home_court))

    # Tug-of-war view: bars grow outward from the centre line toward the side they favour.
    total_pull = sum(abs(v) for _, v in contribs)
    max_pull = max(abs(v) for _, v in contribs)

    rows_html = (
        "<div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>"
        "<div style='flex:1;display:flex;flex-direction:column;align-items:center;gap:6px'>"
        f"<img src='{logo(home)}' width='72'>"
        f"<span style='font-size:17px;font-weight:700;color:{HOME_COLOR}'>{home_name} (home)</span>"
        "</div>"
        "<div style='width:170px'></div>"
        "<div style='flex:1;display:flex;flex-direction:column;align-items:center;gap:6px'>"
        f"<img src='{logo(away)}' width='72'>"
        f"<span style='font-size:17px;font-weight:700;color:{AWAY_COLOR}'>{away_name} (away)</span>"
        "</div>"
        "</div>"
    )
    for label, v in contribs:
        width = abs(v) / max_pull * 100
        strength = (f"<span style='font-size:14px;opacity:0.8;margin:0 8px;"
                    f"white-space:nowrap'>{abs(v) / total_pull * 100:.0f}%</span>")
        bar = (f"<div style='width:{width:.0f}%;height:28px;border-radius:5px;"
               f"background:{HOME_COLOR if v >= 0 else AWAY_COLOR}'></div>")
        if v >= 0:
            left = f"<div style='flex:1;display:flex;align-items:center;justify-content:flex-end'>{strength}{bar}</div>"
            right = "<div style='flex:1'></div>"
        else:
            left = "<div style='flex:1'></div>"
            right = f"<div style='flex:1;display:flex;align-items:center'>{bar}{strength}</div>"
        rows_html += (
            "<div style='display:flex;align-items:center;gap:12px;margin:12px 0'>"
            f"{left}"
            f"<div style='width:170px;text-align:center;font-size:15px;font-weight:600;"
            f"border-left:1px solid #88888855;border-right:1px solid #88888855'>{FRIENDLY_NAMES[label]}</div>"
            f"{right}"
            "</div>"
        )
    st.markdown(rows_html, unsafe_allow_html=True)

    top_label, top_v = max(contribs, key=lambda c: abs(c[1]))
    st.caption(f"Strongest pull: **{FRIENDLY_NAMES[top_label]}**, toward **{home_name if top_v >= 0 else away_name}**.")

    # Team form entering the game: the rolling-average factor values behind the differentials.
    form = load_form()
    if game["GAME_ID"] in form.index:
        f = form.loc[game["GAME_ID"]]
        form_table = pd.DataFrame(
            {
                f"{home_name} (home)": [f[f"home_{suffix}"] for suffix in FORM_COLS.values()],
                f"{away_name} (away)": [f[f"away_{suffix}"] for suffix in FORM_COLS.values()],
            },
            index=list(FORM_COLS),
        )
        st.markdown("**Team form entering the game** (rolling averages, prior games only)")
        st.dataframe(form_table.style.format("{:.3f}"), width="stretch")

    # Reveal.
    if st.button("Reveal actual result"):
        actual_winner = home_name if game["actual_home_win"] == 1 else away_name
        st.write(f"Final result: **{actual_winner} won** "
                 f"({home_name} {'won' if game['actual_home_win'] == 1 else 'lost'} at home).")
        if game["correct"] == 1:
            st.success("Prediction was CORRECT.")
        else:
            st.error("Prediction was incorrect.")
    else:
        st.info("Result hidden — reveal it after reading the prediction.")

# =====================================================================
# Tab 2 — Season performance
# =====================================================================
with tab_perf:
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

    chart_left, chart_right = st.columns(2)

    # Cumulative accuracy across the season: model vs always-pick-home baseline.
    with chart_left:
        st.markdown("**Accuracy over the season**")
        st.caption("Cumulative share of games predicted correctly, in chronological order.")
        cum_model = preds["correct"].expanding().mean()
        cum_home = preds["actual_home_win"].expanding().mean()

        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        ax2.plot(preds["Date"], cum_model * 100, color=HOME_COLOR, label="Model")
        ax2.plot(preds["Date"], cum_home * 100, color=NEUTRAL_COLOR, linestyle="--",
                 label="Always pick home")
        ax2.axhline(50, color="black", linewidth=0.6, alpha=0.5)
        ax2.set_ylabel("Cumulative accuracy (%)")
        ax2.legend(frameon=False, fontsize=9)
        fig2.autofmt_xdate()
        fig2.tight_layout()
        st.pyplot(fig2)

    # Accuracy by model confidence: how often the favoured side wins at each confidence level.
    with chart_right:
        st.markdown("**Accuracy by model confidence**")
        st.caption(
            "Games grouped by the win probability given to the favoured team. Confident "
            "predictions should be right more often than borderline ones."
        )
        confidence = preds["p_home_win"].where(preds["p_home_win"] >= 0.5,
                                               1 - preds["p_home_win"])
        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.00]
        bin_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
        banded = preds.assign(conf_band=pd.cut(confidence, bins, labels=bin_labels))
        by_conf = banded.groupby("conf_band", observed=False)["correct"]
        conf_acc = by_conf.mean() * 100
        conf_n = by_conf.count()

        fig3, ax3 = plt.subplots(figsize=(6, 3.2))
        bars3 = ax3.bar(bin_labels, conf_acc, color=HOME_COLOR)
        for bar, acc, n in zip(bars3, conf_acc, conf_n):
            ax3.annotate(f"{acc:.0f}%\n(n={n})", xy=(bar.get_x() + bar.get_width() / 2, acc),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax3.axhline(50, color="black", linewidth=0.6, alpha=0.5)
        ax3.set_ylim(0, 100)
        ax3.set_xlabel("Win probability given to the favoured team")
        ax3.set_ylabel("Accuracy (%)")
        fig3.tight_layout()
        st.pyplot(fig3)
