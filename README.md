# Football Match Prediction System

## Project Overview

This project develops a machine learning system for predicting football match outcomes using historical international football match data.

The system considers home and away team performance along with team-strength and recent-form features. Machine learning models are trained on historical matches and evaluated u
# FIFA World Cup Streamlit App

This app is built from the final production pipeline in `football_match_prediction(1).ipynb`.

## Model recovered from the notebook

The final production model is the **time-based neutral-venue Random Forest**:

- Model object: `best_rf_time`
- Features:
  - `elo_diff`
  - `fifa_diff`
  - `form_diff_goals_conceded`
  - `form_diff_goals_scored`
  - `form_diff_win_ratio`
- Recent-form window: 5 matches
- Target classes: `Draw`, `Team A Win`, `Team B Win`
- Test accuracy in the notebook: about 0.5423
- 2026 fixture comparison in the notebook: 61.90%

Important: the uploaded notebook does **not** contain a `calibrated_clf` / isotonic calibration step. The app therefore uses the final Random Forest that is actually present in the notebook.

## 1. Export the artifacts

Open your notebook and run all cells through the final model/evaluation cells. Then run `save_artifacts.py` as a final cell.

It creates:

```text
artifacts/
    wc_model.joblib
    matches.parquet
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Run Streamlit

```bash
streamlit run app.py
```

## 4. Important note about exact production parity

The notebook's 2026 inference uses a separate fixture file:

`World_Cup_2026_Fixtures_with_Elo_FIFA.xlsx`

The starter app here is for arbitrary Team A vs Team B predictions. It gets each team's latest historical Elo/FIFA values from `matches.parquet` and computes the notebook's 5-match recent form before the selected prediction date.

For exact parity with the notebook's 2026 fixture predictions, the next refinement is to export the 2026 fixture/rating table too and have the app use those fixed 2026 Elo/FIFA values.

## Folder layout

```text
football_streamlit_app/
├── app.py
├── save_artifacts.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── wc_model.joblib
    └── matches.parquet
```
