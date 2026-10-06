# Baseball ABS modeling and research

Source code for challenge likelihood, success conditional on a challenge, catcher review, and supporting baseball research.

This repository contains source code and selected model configuration only. Input datasets, prediction files, fitted models, cached responses, reports, and derived results are supplied separately or kept local. The original assignment document is not included.

## Setup

Python used: 3.14.2.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Question 1: predictions

Place the original provided `data-train.csv` and unfilled `data-test.csv` in `question_1_modeling/` locally. Do not commit them.

```bash
python question_1_modeling/run_pipeline.py --evaluate
```

This reproduces game-grouped development validation, fits both selected XGBoost models on eligible training rows, and writes `question_1_modeling/submission/data-test.csv`. All three prediction columns are filled and original test rows are preserved. `final_config.json` contains selected settings; no previous experiment output is required.

To tune both models:

```bash
python question_1_modeling/tune_optuna.py
```

Defaults: 20 challenge trials on five whole-game folds; 30 success trials on three sets of five whole-game folds, using an 80% fitting partition and a separate game holdout. `--model challenge` or `--model success` selects one model. The chosen configuration is followed by final training and submission generation.

Historical feature experiments are in `question_1_modeling/experiments/`. Some require the corresponding earlier local experiment outputs. They are not needed by the final pipeline.

Validation folds and the success holdout were used during development; these are not untouched external estimates. Challenge source follows call eligibility: pitching team for a ball, hitting team for a called strike. Success fitting uses challenged pitches only; probabilities on unchallenged pitches are extrapolations.

## Question 2: catcher review

`question_2_catcher_review/` contains chart, export, pitch-clustering, and tendency-analysis code. Public examples use the placeholder game `00000000` and catcher `EXAMPLE_CATCHER`; replace these consistently with your local target and supply private game data and model artifacts. The exact game-specific report and source version are in the private submission package.

Pitch families use standardized velocity, handedness-normalized horizontal movement, and vertical movement. Labels are broad cluster interpretations, not verified pitch types. Defensive challenge labels do not establish whether catcher or pitcher initiated a challenge.

## Question 3: supporting research

`question_3_research/exploration/` contains FanGraphs discipline persistence, configurable ISO decline, and half-season zone comparisons. The main study downloads public leaderboards when absent. The zone comparison requires locally supplied `raw_h1.json` and `raw_h2.json` in its output directory; public responses are not redistributed here.

These analyses are descriptive. For an assessment of a historical organizational decision, use only information available when that decision was made. Written applicant responses are supplied separately.
