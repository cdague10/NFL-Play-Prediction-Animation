# NFL Live Play Animation + xPass Projection

This project builds a live NFL play animation and predicts run/pass probability (xPass) before and during each play.

It combines:
- Situation-based xPass from `plays.csv` context (down, distance, field position, clock, score, formation, alignment)
- Visual xPass from tracking geometry and player movement
- A blended projection shown frame-by-frame in the animation

## Visual Demo (.mp4)

<video src="play_2022092502_738_animation.mp4" controls width="900"></video>

[play_2022092502_738_animation.mp4](play_2022092502_738_animation.mp4)

## What The Project Predicts

The output is a pass probability (`P(pass)`) and a corresponding run probability (`1 - P(pass)`):
- Before snap: baseline projection using situation and pre-snap alignment/spacing
- Live frames: updated projection as players move after snap

## xPass Components

### 1) Situation-Based xPass (Context Model)

The situation model uses only pre-snap context from `plays.csv`:
- down
- yardsToGo
- quarter
- absoluteYardlineNumber
- gameClockSeconds (derived from `gameClock`)
- scoreDiff
- offenseFormation
- receiverAlignment

Model type:
- `GradientBoostingClassifier` inside a preprocessing pipeline
- Numeric imputation (median)
- Categorical imputation + one-hot encoding

Target labeling:
- Pass if `passResult` in {C, I, IN, S} or `isDropback` is true
- Run if `passResult == R` or rush location indicates run
- QB spikes/kneels removed

This gives a pre-snap, situation-only pass probability.

### 2) Visual xPass (Tracking-Based)

Two tracking-driven models are trained:
- Coordinate pre-snap model (final before-snap frame)
- Live frame model (frame-by-frame from around snap through early play development)

Visual features are built from standardized coordinates (`x_std`, `y_std`) and include:
- Formation width/depth spread (`pre_width_std`, `pre_depth_std`)
- Offensive speed summaries (`pre_speed_mean`, `pre_speed_max`, live speed stats)
- "Inner 9" structure features (line interior spacing and depth behavior)
- Ball-relative depth/lateral offsets
- Displacement from early pre-snap reference positions
- Condensed alignment flag from receiver alignment text
- Frame index relative to snap (`frame_from_snap`) for live model timing

Model type:
- `GradientBoostingClassifier` + median imputation

This yields a visual pass probability from formation geometry and motion cues.

## How The Final Projection Is Created

The final frame-level projection is a weighted blend:

`combined_pass_prob = clip(0.25 * plays_model_prob + 0.35 * coord_pre_prob + 0.40 * live_prob, 0.02, 0.98)`

Where:
- `plays_model_prob` = situation-based xPass from `plays.csv`
- `coord_pre_prob` = pre-snap visual xPass from tracking alignment features
- `live_prob` = live visual xPass for the current frame
- `clip(..., 0.02, 0.98)` keeps probabilities bounded for stability/visual readability

For frames without live features, the code falls back to a pre-play blend:

`pre_combined = clip(0.25 * plays_model_prob + 0.75 * coord_pre_prob, 0.02, 0.98)`

## Packages

Core Python packages used:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `ipython`
- `jupyter`
- `pyarrow` (Parquet support)

Install with:

```bash
pip install -r requirements.txt
```

## Platforms

Tested/targeted environment:
- Python 3.10+
- Windows (works on macOS/Linux as well)
- Jupyter Notebook or VS Code Notebook support

## What You Need To Run

1. Python 3.10+ installed
2. Dependencies installed from `requirements.txt`
3. Local data files in the project root:
   - `plays.csv`
   - `tracking_week_1.csv` ... `tracking_week_9.csv`

> Note: These Data Bowl source files are intentionally excluded from version control via `.gitignore`.

## Parquet Workflow (Recommended)

To speed up loading large tracking weeks, convert each CSV once:

```bash
c:/Users/Connor/Documents/NFL_Play_Animation/.venv/Scripts/python.exe convert_tracking_to_parquet.py
```

After conversion, the project now prefers `tracking_week_*.parquet` automatically and falls back to CSV when Parquet is missing.

## Run Instructions

### Notebook flow

1. Open `NFL_play_animation.ipynb`
2. Run the model training cells
3. Run the animation cell for your selected game/play/week

### Script usage

`live_play_prob.py` exposes `run_live_play_animation(...)` which returns embeddable HTML for the animated visualization.

## GitHub Deployment Steps

From this folder:

```bash
git init
git add .
git commit -m "Initial commit: NFL live animation + xPass"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If the remote already exists, skip `git remote add` and run:

```bash
git remote set-url origin https://github.com/<your-username>/<your-repo>.git
```

## Data Source Note

Training data comes from NFL Big Data Bowl 2026 datasets. This repository excludes the raw CSV source files; users should add them locally in the project root before running.
