
import os
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "wc_model.joblib"
MATCHES_PATH = ARTIFACT_DIR / "matches.parquet"
RATINGS_PATH = ARTIFACT_DIR / "team_ratings.parquet"


st.set_page_config(
    page_title="World Cup Match Predictor",
    page_icon="⚽",
    layout="wide",
)


# -------------------------------------------------------------------
# Team-name standardization copied from the notebook's inference logic.
# -------------------------------------------------------------------
MANUAL_TEAM_MAPPING = {
    "south korea": "south korea",
    "korea republic": "south korea",
    "united states": "united states",
    "usa": "united states",
    "united states virgina islands": "united states",
    "ivory coast": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "iran": "iran",
    "ir iran": "iran",
    "saudi arabia": "saudi arabia",
    "czech republic": "czech republic",
    "czechia": "czech republic",
    "new zealand": "new zealand",
    "aotearoa new zealand": "new zealand",
    "south africa": "south africa",
    "bosnia and herzegovina": "bosnia and herzegovina",
    "curacao": "curacao",
    "cape verde": "cape verde",
    "dr congo": "dr congo",
    "congo dr": "dr congo",
}


def robust_standardize(name):
    if not isinstance(name, str):
        return name
    name = name.replace("â ", " ")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = name.lower()
    cleaned = "".join(ch for ch in name if ch.isalnum())
    for official_name, standardized in MANUAL_TEAM_MAPPING.items():
        key = "".join(ch for ch in official_name.lower() if ch.isalnum())
        if cleaned == key:
            return standardized
    return " ".join(name.split())


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Create the model artifact using save_artifacts.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_matches():
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MATCHES_PATH}. Export matches_ml to Parquet first."
        )
    df = pd.read_parquet(MATCHES_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


@st.cache_data
def load_ratings():
    if not RATINGS_PATH.exists():
        return None
    df = pd.read_parquet(RATINGS_PATH)
    if "team" in df.columns:
        df["team"] = df["team"].apply(robust_standardize)
    return df


def get_team_recent_form(team_name, match_date, df_history, window=5):
    """Same recent-form definition used in the notebook."""
    team_matches = df_history[
        ((df_history["home_team"] == team_name) | (df_history["away_team"] == team_name))
        & (df_history["match_date"] < match_date)
    ].sort_values("match_date", ascending=False).head(window)

    if team_matches.empty:
        return {
            "avg_goals_scored": 0.0,
            "avg_goals_conceded": 0.0,
            "win_ratio": 0.0,
        }

    goals_scored = []
    goals_conceded = []
    results = []

    for _, row in team_matches.iterrows():
        if row["home_team"] == team_name:
            goals_scored.append(float(row["home_goals"]))
            goals_conceded.append(float(row["away_goals"]))
            if row["match_result"] == "Home Win":
                results.append("win")
            elif row["match_result"] == "Draw":
                results.append("draw")
            else:
                results.append("loss")
        else:
            goals_scored.append(float(row["away_goals"]))
            goals_conceded.append(float(row["home_goals"]))
            if row["match_result"] == "Away Win":
                results.append("win")
            elif row["match_result"] == "Draw":
                results.append("draw")
            else:
                results.append("loss")

    return {
        "avg_goals_scored": float(np.mean(goals_scored)),
        "avg_goals_conceded": float(np.mean(goals_conceded)),
        "win_ratio": results.count("win") / len(results),
    }


def latest_team_rating(team_name, matches, rating_col_home, rating_col_away):
    """Use the latest historical pre-match rating available for a team."""
    home = (
        matches.loc[matches["home_team"] == team_name, ["match_date", rating_col_home]]
        .rename(columns={rating_col_home: "rating"})
    )
    away = (
        matches.loc[matches["away_team"] == team_name, ["match_date", rating_col_away]]
        .rename(columns={rating_col_away: "rating"})
    )
    both = pd.concat([home, away], ignore_index=True).dropna(subset=["rating"])
    if both.empty:
        return np.nan
    return float(both.sort_values("match_date").iloc[-1]["rating"])


def make_feature_row(team_a, team_b, match_date, matches, train_means):
    team_a = robust_standardize(team_a)
    team_b = robust_standardize(team_b)

    # Ratings: use the latest available historical values.
    fifa_a = latest_team_rating(team_a, matches, "fifa_points_home", "fifa_points_away")
    fifa_b = latest_team_rating(team_b, matches, "fifa_points_home", "fifa_points_away")
    elo_a = latest_team_rating(team_a, matches, "elo_rating_home", "elo_rating_away")
    elo_b = latest_team_rating(team_b, matches, "elo_rating_home", "elo_rating_away")

    # Form is strictly based on matches before the prediction date.
    form_a = get_team_recent_form(team_a, match_date, matches, window=5)
    form_b = get_team_recent_form(team_b, match_date, matches, window=5)

    row = pd.DataFrame([{
        "elo_diff": elo_a - elo_b if pd.notna(elo_a) and pd.notna(elo_b) else np.nan,
        "fifa_diff": fifa_a - fifa_b if pd.notna(fifa_a) and pd.notna(fifa_b) else np.nan,
        "form_diff_goals_conceded": (
            form_a["avg_goals_conceded"] - form_b["avg_goals_conceded"]
        ),
        "form_diff_goals_scored": (
            form_a["avg_goals_scored"] - form_b["avg_goals_scored"]
        ),
        "form_diff_win_ratio": form_a["win_ratio"] - form_b["win_ratio"],
    }])

    # Match the notebook: use training-set means for imputation.
    row = row.fillna(pd.Series(train_means))
    return row, {
        "fifa_a": fifa_a,
        "fifa_b": fifa_b,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "form_a": form_a,
        "form_b": form_b,
    }


# -------------------------------------------------------------------
# Load artifacts
# -------------------------------------------------------------------
try:
    artifact = load_model()
    matches = load_matches()
except Exception as exc:
    st.error(str(exc))
    st.stop()

model = artifact["model"]
feature_names = artifact["feature_names"]
train_means = pd.Series(artifact.get("train_means", {}))
metrics = artifact.get("metrics", {})
trained_through = artifact.get("trained_through", "Unknown")

# Safety check: the app should use exactly the saved feature order.
expected_features = [
    "elo_diff",
    "fifa_diff",
    "form_diff_goals_conceded",
    "form_diff_goals_scored",
    "form_diff_win_ratio",
]
if feature_names != expected_features:
    st.error(
        "Feature mismatch between the saved model and this app. "
        f"Saved: {feature_names}; expected: {expected_features}"
    )
    st.stop()

# Some older artifacts may not contain training means; require them for exact reproduction.
if train_means.empty:
    st.error(
        "The model artifact does not contain training-set imputation means. "
        "Recreate it using the supplied save_artifacts.py."
    )
    st.stop()

# The LabelEncoder classes are stored in the artifact.
classes = np.array(artifact.get("classes", ["Draw", "Team A Win", "Team B Win"]))


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("⚽ FIFA World Cup Match Predictor")
st.caption(
    "Neutral-venue prediction: Team A and Team B are ordering conventions, "
    "not home/away designations."
)

with st.sidebar:
    st.header("Model information")
    st.write(f"**Trained through:** {trained_through}")
    if "test_acc" in metrics:
        st.write(f"**Test accuracy:** {metrics['test_acc']:.2%}")
    if "macro_f1" in metrics:
        st.write(f"**Test macro F1:** {metrics['macro_f1']:.3f}")
    st.write("**Model:** Time-based Random Forest")
    st.write("**Recent form window:** 5 matches")
    st.write("**Features:** Elo, FIFA points, recent form")

tabs = st.tabs(["Match Prediction", "Model Details", "Data"])

with tabs[0]:
    teams = sorted(
        set(matches["home_team"].dropna().unique())
        | set(matches["away_team"].dropna().unique())
    )

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=0)
    with col2:
        default_b = 1 if len(teams) > 1 else 0
        team_b = st.selectbox("Team B", teams, index=default_b)

    prediction_date = st.date_input(
        "Prediction date",
        value=pd.Timestamp("2026-06-11").date(),
        min_value=pd.Timestamp("2025-01-01").date(),
    )

    if team_a == team_b:
        st.warning("Please select two different teams.")
    elif st.button("Predict Match", type="primary", use_container_width=True):
        date = pd.Timestamp(prediction_date)
        X_new, details = make_feature_row(
            team_a, team_b, date, matches, train_means
        )

        encoded_prediction = model.predict(X_new)[0]
        label = classes[int(encoded_prediction)]

        st.subheader(f"{team_a}  vs  {team_b}")
        if label == "Team A Win":
            headline = f"Predicted result: {team_a} Win"
        elif label == "Team B Win":
            headline = f"Predicted result: {team_b} Win"
        else:
            headline = "Predicted result: Draw"
        st.success(headline)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_new)[0]
            prob_df = pd.DataFrame({
                "Outcome": [
                    f"{team_a} Win",
                    "Draw",
                    f"{team_b} Win",
                ],
                "Probability": [
                    float(probabilities[np.where(classes == "Team A Win")[0][0]])
                    if "Team A Win" in classes else np.nan,
                    float(probabilities[np.where(classes == "Draw")[0][0]])
                    if "Draw" in classes else np.nan,
                    float(probabilities[np.where(classes == "Team B Win")[0][0]])
                    if "Team B Win" in classes else np.nan,
                ],
            })
            st.bar_chart(prob_df.set_index("Outcome"))
            st.dataframe(
                prob_df.style.format({"Probability": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Features used for this prediction"):
            feature_display = X_new.T.rename(columns={0: "Value"})
            st.dataframe(feature_display, use_container_width=True)

        with st.expander("Team context"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{team_a}**")
                st.write(f"Elo: {details['elo_a']:.1f}" if pd.notna(details["elo_a"]) else "Elo: unavailable")
                st.write(f"FIFA points: {details['fifa_a']:.1f}" if pd.notna(details["fifa_a"]) else "FIFA points: unavailable")
                st.write(details["form_a"])
            with c2:
                st.markdown(f"**{team_b}**")
                st.write(f"Elo: {details['elo_b']:.1f}" if pd.notna(details["elo_b"]) else "Elo: unavailable")
                st.write(f"FIFA points: {details['fifa_b']:.1f}" if pd.notna(details["fifa_b"]) else "FIFA points: unavailable")
                st.write(details["form_b"])


with tabs[1]:
    st.subheader("Final production features")
    st.write(feature_names)

    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": imp,
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature"))

    if metrics:
        st.subheader("Saved evaluation metrics")
        st.json(metrics)


with tabs[2]:
    st.subheader("Historical match data")
    st.write(f"Rows: {len(matches):,}")
    st.write(
        f"Period: {matches['match_date'].min().date()} → "
        f"{matches['match_date'].max().date()}"
    )
    st.dataframe(matches.head(50), use_container_width=True)
