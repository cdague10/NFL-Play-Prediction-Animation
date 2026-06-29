from pathlib import Path
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tracking_data_io import resolve_tracking_files, read_tracking_file, read_tracking_keys, week_num_from_name

def _week_num_from_name(path_obj):
    return week_num_from_name(path_obj)

def _ordinal_down(value):
    mapping = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    if pd.isna(value):
        return ""
    return mapping.get(int(value), f"{int(value)}th")

def _label_pass_run(df):
    out = df.copy()
    out["is_pass"] = np.nan
    if "passResult" in out.columns:
        pr = out["passResult"].astype(str).str.upper().str.strip()
        out.loc[pr.isin(["C", "I", "IN", "S"]), "is_pass"] = 1
        out.loc[pr.eq("R"), "is_pass"] = 0
    if "isDropback" in out.columns:
        db = out["isDropback"].astype(str).str.upper().str.strip()
        out.loc[(db == "TRUE") | (db == "1"), "is_pass"] = 1
    if "rushLocationType" in out.columns:
        out.loc[out["rushLocationType"].notna(), "is_pass"] = 0
    if "qbSpike" in out.columns:
        out = out.loc[~out["qbSpike"].astype(str).str.upper().eq("TRUE")].copy()
    if "qbKneel" in out.columns:
        out = out.loc[~out["qbKneel"].eq(1)].copy()
    out = out.loc[out["is_pass"].isin([0, 1])].copy()
    out["is_pass"] = out["is_pass"].astype(int)
    return out


def _inner9_features(group):
    g = group.sort_values("y_std")
    if len(g) > 9:
        g = g.iloc[1:-1]
    out = {
        "inner9_width_std": float(g["y_std"].std()) if len(g) > 1 else 0.0,
        "inner9_depth_std": float(g["x_std"].std()) if len(g) > 1 else 0.0,
        "inner9_speed_mean": float(g["s"].mean()) if len(g) else 0.0,
    }
    if "ball_x" in g.columns:
        depth = g["ball_x"] - g["x_std"]
        out["inner9_depth_mean_from_ball"] = float(depth.mean()) if len(depth) else 0.0
        out["inner9_backfield_count"] = int((depth > 1.0).sum()) if len(depth) else 0
    else:
        out["inner9_depth_mean_from_ball"] = 0.0
        out["inner9_backfield_count"] = 0
    return pd.Series(out)


def _clock_to_sec(v):
    try:
        m, s = str(v).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan


def _parse_jersey_number(series):
    # Normalize jersey values that may come in as float, int, or string.
    return pd.to_numeric(series, errors="coerce")


def _drop_offensive_lineman_rows(df):
    jersey_num = _parse_jersey_number(df.get("jerseyNumber"))
    is_ol_jersey = jersey_num.between(50, 79, inclusive="both")
    return df.loc[~is_ol_jersey].copy()


def run_live_play_animation(
    play_id_input,
    game_id_input,
    week_input,
    predict_pass_prob_fn=None,
    max_train_plays=3000,
    live_max_frames_per_play=14,
    live_frame_window=20,
    random_state=42,
):
    tracking_candidates = resolve_tracking_files(Path("."))
    if not tracking_candidates:
        raise FileNotFoundError("No tracking_week_*.parquet or tracking_week_*.csv files found.")

    key_parts = []
    for path in tracking_candidates:
        wk = _week_num_from_name(path)
        keys = read_tracking_keys(path)
        keys["tracking_week"] = wk
        key_parts.append(keys)
    all_tracking_keys = pd.concat(key_parts, ignore_index=True).drop_duplicates()

    matches = all_tracking_keys.loc[all_tracking_keys["playId"] == int(play_id_input)].copy()
    if game_id_input is not None:
        matches = matches.loc[matches["gameId"] == int(game_id_input)]
    if week_input is not None:
        matches = matches.loc[matches["tracking_week"] == int(week_input)]
    if matches.empty:
        raise ValueError("No tracking play match found for the inputs.")

    chosen = matches.sort_values(["tracking_week", "gameId", "playId"]).iloc[0]
    anim_game_id = int(chosen["gameId"])
    anim_play_id = int(chosen["playId"])
    anim_week = int(chosen["tracking_week"])

    plays_path = Path("plays.csv")
    if not plays_path.exists():
        raise FileNotFoundError("plays.csv is required.")

    plays_raw = pd.read_csv(plays_path)
    plays_labeled = _label_pass_run(plays_raw)

    train_cols = [
        "gameId", "playId", "is_pass", "possessionTeam", "down", "yardsToGo", "quarter", "receiverAlignment"
    ]
    train_cols = [c for c in train_cols if c in plays_labeled.columns]
    train_keys = plays_labeled[train_cols].copy()
    train_keys = train_keys.merge(all_tracking_keys[["gameId", "playId"]].drop_duplicates(), on=["gameId", "playId"], how="inner")
    if len(train_keys) > max_train_plays:
        train_keys = train_keys.sample(max_train_plays, random_state=random_state)

    key_set = set(zip(train_keys["gameId"].to_numpy(), train_keys["playId"].to_numpy()))
    tcols = ["gameId", "playId", "nflId", "frameId", "frameType", "club", "playDirection", "x", "y", "s", "event", "jerseyNumber"]
    parts = []
    for p in tracking_candidates:
        part = read_tracking_file(p, columns=tcols)
        part = part.loc[part[["gameId", "playId"]].apply(tuple, axis=1).isin(key_set)]
        if not part.empty:
            parts.append(part)
    tracking_train = pd.concat(parts, ignore_index=True)

    tracking_train["playDirection"] = tracking_train["playDirection"].astype(str).str.lower()
    tracking_train["x_std"] = np.where(tracking_train["playDirection"].eq("left"), 120 - tracking_train["x"], tracking_train["x"])
    tracking_train["y_std"] = np.where(tracking_train["playDirection"].eq("left"), 53.3 - tracking_train["y"], tracking_train["y"])
    tracking_train = tracking_train.merge(train_keys, on=["gameId", "playId"], how="left")

    snap_df = (
        tracking_train.loc[tracking_train["event"].eq("ball_snap")]
        .groupby(["gameId", "playId"], as_index=False)["frameId"]
        .min()
        .rename(columns={"frameId": "snap_frame"})
    )
    tracking_train = tracking_train.merge(snap_df, on=["gameId", "playId"], how="left")
    tracking_train["frame_from_snap"] = tracking_train["frameId"] - tracking_train["snap_frame"]

    ball_frame = tracking_train.loc[tracking_train["club"].eq("football"), ["gameId", "playId", "frameId", "x_std", "y_std"]].rename(columns={"x_std": "ball_x", "y_std": "ball_y"})
    off_frame = tracking_train.loc[tracking_train["club"] == tracking_train["possessionTeam"]].copy()
    off_frame = _drop_offensive_lineman_rows(off_frame)
    off_frame = off_frame.merge(ball_frame, on=["gameId", "playId", "frameId"], how="left")

    pre_last = (
        tracking_train.loc[tracking_train["frameType"].eq("BEFORE_SNAP")]
        .groupby(["gameId", "playId"], as_index=False)["frameId"]
        .max().rename(columns={"frameId": "pre_last_frame"})
    )
    pre_rows = off_frame.merge(pre_last, on=["gameId", "playId"], how="inner")
    pre_rows = pre_rows.loc[pre_rows["frameId"] == pre_rows["pre_last_frame"]].copy()

    pre_base = pre_rows.groupby(["gameId", "playId"], as_index=False).agg(
        pre_width_std=("y_std", "std"),
        pre_depth_std=("x_std", "std"),
        pre_speed_mean=("s", "mean"),
        pre_speed_max=("s", "max"),
    )
    pre_base["pre_width_std"] = pre_base["pre_width_std"].fillna(0.0)
    pre_base["pre_depth_std"] = pre_base["pre_depth_std"].fillna(0.0)
    pre_i9 = pre_rows.groupby(["gameId", "playId"], group_keys=False).apply(_inner9_features).reset_index()

    align = train_keys[["gameId", "playId", "receiverAlignment", "down", "yardsToGo", "quarter", "is_pass"]].drop_duplicates()
    align["receiverAlignment"] = align["receiverAlignment"].astype(str).str.upper()
    align["is_condensed_align"] = (~align["receiverAlignment"].str.contains("3|4", regex=True)).astype(int)

    pre_model_df = pre_base.merge(pre_i9, on=["gameId", "playId"], how="left").merge(align, on=["gameId", "playId"], how="left").dropna(subset=["is_pass"])
    pre_feature_cols = [
        "pre_width_std", "pre_depth_std", "pre_speed_mean", "pre_speed_max",
        "inner9_width_std", "inner9_depth_std", "inner9_speed_mean",
        "inner9_depth_mean_from_ball", "inner9_backfield_count",
        "is_condensed_align", "down", "yardsToGo", "quarter",
    ]
    pre_feature_cols = [c for c in pre_feature_cols if c in pre_model_df.columns]

    pre_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=260, learning_rate=0.05, max_depth=3, random_state=random_state)),
    ])
    pre_model.fit(pre_model_df[pre_feature_cols], pre_model_df["is_pass"].astype(int))

    live_rows = off_frame.loc[off_frame["frame_from_snap"].between(-2, live_frame_window)].copy()
    live_frames = live_rows[["gameId", "playId", "frameId"]].drop_duplicates().sort_values(["gameId", "playId", "frameId"])
    live_frames["rk"] = live_frames.groupby(["gameId", "playId"]).cumcount() + 1
    live_frames = live_frames.loc[live_frames["rk"] <= live_max_frames_per_play]
    live_rows = live_rows.merge(live_frames[["gameId", "playId", "frameId"]], on=["gameId", "playId", "frameId"], how="inner")

    pre_first = (
        tracking_train.loc[tracking_train["frameType"].eq("BEFORE_SNAP")]
        .groupby(["gameId", "playId"], as_index=False)["frameId"]
        .min().rename(columns={"frameId": "pre_first_frame"})
    )
    pre_first_rows = off_frame.merge(pre_first, on=["gameId", "playId"], how="inner")
    pre_first_rows = pre_first_rows.loc[
        pre_first_rows["frameId"] == pre_first_rows["pre_first_frame"], ["gameId", "playId", "nflId", "x_std", "y_std"]
    ].rename(columns={"x_std": "x0", "y_std": "y0"})

    live_rows = live_rows.merge(pre_first_rows, on=["gameId", "playId", "nflId"], how="left")
    live_rows["disp_from_presnap"] = np.hypot(live_rows["x_std"] - live_rows["x0"], live_rows["y_std"] - live_rows["y0"])
    live_rows["depth_from_ball"] = live_rows["ball_x"] - live_rows["x_std"]
    live_rows["abs_lat_from_ball"] = (live_rows["y_std"] - live_rows["ball_y"]).abs()

    live_base = live_rows.groupby(["gameId", "playId", "frameId"], as_index=False).agg(
        live_speed_mean=("s", "mean"),
        live_speed_max=("s", "max"),
        live_disp_mean=("disp_from_presnap", "mean"),
        live_depth_mean=("depth_from_ball", "mean"),
        live_depth_std=("depth_from_ball", "std"),
        live_lat_mean=("abs_lat_from_ball", "mean"),
    )
    live_base["live_depth_std"] = live_base["live_depth_std"].fillna(0.0)
    live_i9 = live_rows.groupby(["gameId", "playId", "frameId"], group_keys=False).apply(_inner9_features).reset_index()

    live_model_df = live_base.merge(live_i9, on=["gameId", "playId", "frameId"], how="left")
    live_model_df = live_model_df.merge(
        tracking_train[["gameId", "playId", "frameId", "frame_from_snap"]].drop_duplicates(),
        on=["gameId", "playId", "frameId"], how="left",
    )
    live_model_df = live_model_df.merge(align[["gameId", "playId", "is_condensed_align", "is_pass"]], on=["gameId", "playId"], how="left").dropna(subset=["is_pass"])

    live_feature_cols = [
        "frame_from_snap", "live_speed_mean", "live_speed_max", "live_disp_mean",
        "live_depth_mean", "live_depth_std", "live_lat_mean",
        "inner9_width_std", "inner9_depth_std", "inner9_speed_mean",
        "inner9_depth_mean_from_ball", "inner9_backfield_count", "is_condensed_align",
    ]
    live_feature_cols = [c for c in live_feature_cols if c in live_model_df.columns]

    live_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=220, learning_rate=0.05, max_depth=3, random_state=random_state)),
    ])
    live_model.fit(live_model_df[live_feature_cols], live_model_df["is_pass"].astype(int))

    # Selected play scoring
    sel_tracking_path = next((p for p in tracking_candidates if _week_num_from_name(p) == anim_week), None)
    if sel_tracking_path is None:
        raise FileNotFoundError(f"No tracking file found for week {anim_week}.")
    play_df = read_tracking_file(sel_tracking_path, columns=tcols)
    play_df = play_df.loc[(play_df["gameId"] == anim_game_id) & (play_df["playId"] == anim_play_id)].copy()
    play_df["playDirection"] = play_df["playDirection"].astype(str).str.lower()
    play_df["x_std"] = np.where(play_df["playDirection"].eq("left"), 120 - play_df["x"], play_df["x"])
    play_df["y_std"] = np.where(play_df["playDirection"].eq("left"), 53.3 - play_df["y"], play_df["y"])
    play_df = play_df.sort_values(["frameId", "club", "nflId"])
    frames = np.sort(play_df["frameId"].unique())

    meta = plays_raw.loc[(plays_raw["gameId"] == anim_game_id) & (plays_raw["playId"] == anim_play_id)]
    if meta.empty:
        raise ValueError("Selected play missing from plays.csv")
    r = meta.iloc[0]

    context_parts = []
    if pd.notna(r.get("down")) and pd.notna(r.get("yardsToGo")):
        context_parts.append(f"{_ordinal_down(r['down'])} & {int(r['yardsToGo'])}")
    if pd.notna(r.get("quarter")):
        clk = str(r.get("gameClock", "")) if pd.notna(r.get("gameClock")) else ""
        context_parts.append(f"Q{int(r['quarter'])} {clk}".strip())
    if pd.notna(r.get("yardlineSide")) and pd.notna(r.get("yardlineNumber")):
        context_parts.append(f"{r['yardlineSide']} {int(r['yardlineNumber'])}")
    if pd.notna(r.get("offenseFormation")):
        context_parts.append(str(r["offenseFormation"]))
    if pd.notna(r.get("receiverAlignment")):
        context_parts.append(str(r["receiverAlignment"]))
    context_line = " | ".join(context_parts)
    description_line = textwrap.fill(str(r.get("playDescription", "")), width=90)

    pr = str(r.get("passResult", "")).upper().strip()
    rl = str(r.get("rushLocationType", "")).strip()
    if pr in ("C", "I", "IN", "S"):
        play_result_label = "PASS"
    elif pr == "R" or rl not in ("", "NAN", "NONE", "nan"):
        play_result_label = "RUN"
    else:
        play_result_label = "UNKNOWN"

    pos_team = str(r.get("possessionTeam", ""))
    if pos_team in ("", "nan", "None"):
        non_ball = play_df.loc[play_df["club"] != "football", "club"]
        pos_team = non_ball.mode().iat[0] if len(non_ball) else "unknown"

    snap_frame_id = None
    snap_events = play_df.loc[play_df["event"].eq("ball_snap"), "frameId"]
    if not snap_events.empty:
        snap_frame_id = int(snap_events.min())

    ref_los_frame = snap_frame_id if snap_frame_id is not None else int(frames.min())
    ball_ref = play_df.loc[(play_df["frameId"] == ref_los_frame) & (play_df["club"] == "football")]
    los_y = float(ball_ref.iloc[0]["x_std"]) if not ball_ref.empty else None
    first_down_y = None
    if los_y is not None and pd.notna(r.get("yardsToGo")):
        first_down_y = float(np.clip(los_y + float(r["yardsToGo"]), 0, 120))

    plays_model_prob = 0.5
    if predict_pass_prob_fn is not None and pd.notna(r.get("down")) and pd.notna(r.get("yardsToGo")) and pd.notna(r.get("quarter")) and pd.notna(r.get("absoluteYardlineNumber")):
        out = predict_pass_prob_fn(
            down=int(r["down"]),
            yards_to_go=int(r["yardsToGo"]),
            quarter=int(r["quarter"]),
            absolute_yardline=int(r["absoluteYardlineNumber"]),
            offense_formation=str(r.get("offenseFormation", "UNKNOWN")),
            receiver_alignment=str(r.get("receiverAlignment", "UNKNOWN")),
            game_clock_seconds=_clock_to_sec(r.get("gameClock")),
            score_diff=float(r.get("preSnapHomeScore", 0) or 0) - float(r.get("preSnapVisitorScore", 0) or 0),
        )
        plays_model_prob = float(out["pass_probability"])

    # selected play features
    sel_ball = play_df.loc[play_df["club"].eq("football"), ["frameId", "x_std", "y_std"]].rename(columns={"x_std": "ball_x", "y_std": "ball_y"})
    sel_off = play_df.loc[play_df["club"] == pos_team].copy()
    sel_off = _drop_offensive_lineman_rows(sel_off)
    sel_off = sel_off.merge(sel_ball, on="frameId", how="left")
    sel_pre = sel_off.loc[sel_off["frameType"].eq("BEFORE_SNAP")]
    if sel_pre.empty:
        sel_pre = sel_off.copy()

    sel_pre_first = int(sel_pre["frameId"].min())
    base = sel_pre.loc[sel_pre["frameId"] == sel_pre_first, ["nflId", "x_std", "y_std"]].rename(columns={"x_std": "x0", "y_std": "y0"})
    sel_off = sel_off.merge(base, on="nflId", how="left")
    sel_off["disp_from_presnap"] = np.hypot(sel_off["x_std"] - sel_off["x0"], sel_off["y_std"] - sel_off["y0"])
    sel_off["depth_from_ball"] = sel_off["ball_x"] - sel_off["x_std"]
    sel_off["abs_lat_from_ball"] = (sel_off["y_std"] - sel_off["ball_y"]).abs()
    sel_off["frame_from_snap"] = sel_off["frameId"] - (snap_frame_id if snap_frame_id is not None else int(frames.min()))

    sel_base = sel_off.groupby("frameId", as_index=False).agg(
        live_speed_mean=("s", "mean"),
        live_speed_max=("s", "max"),
        live_disp_mean=("disp_from_presnap", "mean"),
        live_depth_mean=("depth_from_ball", "mean"),
        live_depth_std=("depth_from_ball", "std"),
        live_lat_mean=("abs_lat_from_ball", "mean"),
        frame_from_snap=("frame_from_snap", "first"),
    )
    sel_base["live_depth_std"] = sel_base["live_depth_std"].fillna(0.0)
    sel_i9 = sel_off.groupby("frameId", group_keys=False).apply(_inner9_features).reset_index()
    sel_live = sel_base.merge(sel_i9, on="frameId", how="left")
    recv_align = str(r.get("receiverAlignment", "")).upper().strip()
    sel_live["is_condensed_align"] = int(("3" not in recv_align) and ("4" not in recv_align))

    for c in live_feature_cols:
        if c not in sel_live.columns:
            sel_live[c] = 0.0

    live_probs = live_model.predict_proba(sel_live[live_feature_cols])[:, 1]

    sel_pre_last = int(sel_pre["frameId"].max())
    sel_pre_rows = sel_pre.loc[sel_pre["frameId"] == sel_pre_last].copy()
    ball_last = sel_pre.loc[(sel_pre["frameId"] == sel_pre_last) & (sel_pre["club"] == "football")]
    ball_x = float(ball_last.iloc[0]["x_std"]) if not ball_last.empty else np.nan
    sel_pre_rows["ball_x"] = ball_x

    sel_pre_base = pd.DataFrame([{
        "pre_width_std": float(sel_pre_rows["y_std"].std()) if len(sel_pre_rows) > 1 else 0.0,
        "pre_depth_std": float(sel_pre_rows["x_std"].std()) if len(sel_pre_rows) > 1 else 0.0,
        "pre_speed_mean": float(sel_pre_rows["s"].mean()) if len(sel_pre_rows) else 0.0,
        "pre_speed_max": float(sel_pre_rows["s"].max()) if len(sel_pre_rows) else 0.0,
        "is_condensed_align": int(("3" not in recv_align) and ("4" not in recv_align)),
        "down": float(r.get("down", 1)),
        "yardsToGo": float(r.get("yardsToGo", 10)),
        "quarter": float(r.get("quarter", 1)),
    }])
    for k, v in _inner9_features(sel_pre_rows).items():
        sel_pre_base[k] = v
    for c in pre_feature_cols:
        if c not in sel_pre_base.columns:
            sel_pre_base[c] = 0.0

    coord_pre_prob = float(pre_model.predict_proba(sel_pre_base[pre_feature_cols])[:, 1][0])

    sel_live["combined_pass_prob"] = np.clip(
        0.25 * plays_model_prob + 0.35 * coord_pre_prob + 0.40 * live_probs,
        0.02,
        0.98,
    )
    frame_prob_map = dict(zip(sel_live["frameId"].astype(int), sel_live["combined_pass_prob"].astype(float)))
    combined_probs = [float(frame_prob_map.get(int(fid), np.clip(0.25 * plays_model_prob + 0.75 * coord_pre_prob, 0.02, 0.98))) for fid in frames]

    pre_combined = float(np.clip(0.25 * plays_model_prob + 0.75 * coord_pre_prob, 0.02, 0.98))
    pre_label = "PASS" if pre_combined >= 0.5 else "RUN"

    print(f"Selected -> week={anim_week}, gameId={anim_game_id}, playId={anim_play_id}")
    print(f"Coord training plays used: {train_keys[['gameId','playId']].drop_duplicates().shape[0]:,}")
    print(f"Live training frame rows:  {len(live_model_df):,}")
    print(f"Plays model P(pass):      {plays_model_prob:.3f}")
    print(f"Coord pre P(pass):        {coord_pre_prob:.3f}")
    print(f"Live P(pass) mean:        {np.mean(live_probs):.3f}")
    print(f"Combined pre-play:        {pre_combined:.3f} -> {pre_label}")
    print(f"Actual result:            {play_result_label}")

    # Plot
    clubs = [c for c in sorted(play_df["club"].dropna().unique()) if c != "football"]
    color_map = {
        clubs[0]: "#1f77b4" if len(clubs) > 0 else "#1f77b4",
        clubs[1]: "#d62728" if len(clubs) > 1 else "#d62728",
        "football": "#8b4513",
    }

    fig = plt.figure(figsize=(9, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.22, 3.5, 1.0], hspace=0.08)
    ax_info = fig.add_subplot(gs[0])
    ax_field = fig.add_subplot(gs[1])
    ax_prob = fig.add_subplot(gs[2])

    ax_info.set_axis_off()
    pred_color = "#1565c0" if pre_label == "PASS" else "#b71c1c"
    ax_info.text(0.5, 0.90, context_line, transform=ax_info.transAxes, ha="center", va="top", fontsize=11, fontweight="bold")
    ax_info.text(0.5, 0.54, description_line, transform=ax_info.transAxes, ha="center", va="top", fontsize=8.5, color="#333333", wrap=True)
    ax_info.text(
        0.01,
        0.05,
        f"Pre-play combined: {pre_label} ({pre_combined:.1%} pass) | Plays ML: {plays_model_prob:.1%} | Coord pre ML: {coord_pre_prob:.1%}",
        transform=ax_info.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=pred_color,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffde7", edgecolor=pred_color, linewidth=1.2),
    )

    actual_color = "#1565c0" if play_result_label == "PASS" else "#b71c1c"

    ax_field.set_xlim(0, 53.3)
    ax_field.set_ylim(0, 120)
    ax_field.set_facecolor("#2c7a2c")
    ax_field.add_patch(plt.Rectangle((0, 0), 53.3, 10, color="#1f4f1f", alpha=0.85))
    ax_field.add_patch(plt.Rectangle((0, 110), 53.3, 10, color="#1f4f1f", alpha=0.85))
    for yl in range(0, 121, 5):
        lw = 1.6 if yl % 10 == 0 else 1.0
        al = 0.65 if yl % 10 == 0 else 0.35
        ax_field.plot([0, 53.3], [yl, yl], color="white", lw=lw, alpha=al)
    ax_field.set_xticks([])
    ax_field.set_yticks([])
    for sp in ax_field.spines.values():
        sp.set_visible(False)

    los_artist = None
    first_down_artist = None
    if los_y is not None:
        los_artist = ax_field.axhline(los_y, color="#1e88e5", linewidth=2.8, alpha=0.95, zorder=5)
        ax_field.text(0.4, los_y + 0.8, "LOS", color="#1e88e5", fontsize=9, fontweight="bold", zorder=6)
    if first_down_y is not None:
        first_down_artist = ax_field.axhline(first_down_y, color="#fdd835", linewidth=2.8, alpha=0.95, zorder=5)
        ax_field.text(0.4, first_down_y + 0.8, "1st Down", color="#fdd835", fontsize=9, fontweight="bold", zorder=6)

    scatter_by_club = {}
    for club in clubs + ["football"]:
        sz = 130 if club != "football" else 65
        scatter_by_club[club] = ax_field.scatter([], [], s=sz, c=color_map.get(club, "white"), edgecolors="black", linewidths=0.6, zorder=3)

    frame_label = ax_field.text(0.5, -0.01, "", transform=ax_field.transAxes, ha="center", va="top", fontsize=9, color="#333333")
    snap_line = ax_field.axhline(y=-999, color="white", linewidth=1.8, linestyle="--", alpha=0.9, zorder=4)

    ax_prob.set_xlim(frames.min(), frames.max())
    ax_prob.set_ylim(0, 1)
    ax_prob.set_facecolor("#f5f5f5")
    ax_prob.set_xlabel("Frame", fontsize=9)
    ax_prob.set_ylabel("P(pass)", fontsize=9)
    ax_prob.set_title("Live run/pass probability (frame-varying)", fontsize=9)
    ax_prob.axhspan(0.5, 1.0, alpha=0.07, color="#1565c0")
    ax_prob.axhspan(0.0, 0.5, alpha=0.07, color="#b71c1c")
    ax_prob.axhline(0.5, color="grey", lw=0.8, linestyle=":")

    ax_prob.plot(frames, [plays_model_prob] * len(frames), color="#6d4c41", linewidth=1.4, linestyle="--", label=f"Plays ML: {plays_model_prob:.1%}")
    ax_prob.plot(frames, [coord_pre_prob] * len(frames), color="#00897b", linewidth=1.4, linestyle="--", label=f"Coord pre ML: {coord_pre_prob:.1%}")
    ax_prob.plot(frames, combined_probs, color=pred_color, linewidth=2.8, label="Combined live")
    if snap_frame_id is not None:
        ax_prob.axvline(snap_frame_id, color="black", lw=1.2, alpha=0.8, label="snap")

    ax_prob.text(0.995, 0.98, f"Actual: {play_result_label}", transform=ax_prob.transAxes, ha="right", va="top", fontsize=10, color=actual_color, fontweight="bold")

    current_dot = ax_prob.scatter([], [], color="black", s=38, zorder=5)
    ax_prob.legend(fontsize=8, loc="lower left")
    ax_prob.tick_params(labelsize=8)

    label_artists = []

    def init_anim():
        for sc in scatter_by_club.values():
            sc.set_offsets(np.empty((0, 2)))
        frame_label.set_text("")
        current_dot.set_offsets(np.empty((0, 2)))
        for t in label_artists:
            t.remove()
        label_artists.clear()
        return [*scatter_by_club.values(), frame_label, current_dot]

    def update_anim(i):
        fid = int(frames[i])
        frame = play_df.loc[play_df["frameId"] == fid]

        for t in label_artists:
            t.remove()
        label_artists.clear()

        for club, sc in scatter_by_club.items():
            sub = frame.loc[frame["club"] == club]
            if sub.empty:
                sc.set_offsets(np.empty((0, 2)))
            else:
                sc.set_offsets(sub[["y_std", "x_std"]].to_numpy())

        for club in clubs:
            sub = frame.loc[frame["club"] == club]
            for _, row in sub.iterrows():
                jn = str(row.get("jerseyNumber", "")).replace(".0", "").strip()
                if jn:
                    t = ax_field.text(float(row["y_std"]), float(row["x_std"]), jn, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=4)
                    label_artists.append(t)

        if snap_frame_id is not None and fid >= snap_frame_id:
            snap_rows = play_df.loc[(play_df["frameId"] == snap_frame_id) & (play_df["club"] == "football")]
            if not snap_rows.empty:
                snap_y = float(snap_rows.iloc[0]["x_std"])
                snap_line.set_ydata([snap_y, snap_y])
        else:
            snap_line.set_ydata([-999, -999])

        event = frame["event"].dropna()
        ev_str = f" | {event.iloc[0]}" if len(event) else ""
        ftype = frame["frameType"].iloc[0] if len(frame) else ""
        p_now = combined_probs[i]
        pred_now = "PASS" if p_now >= 0.5 else "RUN"
        frame_label.set_text(f"Frame {fid} ({ftype}){ev_str} | Live: {pred_now} ({p_now:.1%} pass)")

        current_dot.set_offsets(np.array([[fid, p_now]]))

        artists = [*scatter_by_club.values(), frame_label, current_dot, snap_line, *label_artists]
        if los_artist is not None:
            artists.append(los_artist)
        if first_down_artist is not None:
            artists.append(first_down_artist)
        return artists

    ani_out = animation.FuncAnimation(
        fig,
        update_anim,
        frames=len(frames),
        init_func=init_anim,
        interval=90,
        blit=False,
        repeat=False,
    )

    fig.subplots_adjust(top=0.98, bottom=0.04, hspace=0.12)
    plt.close(fig)
    return HTML(ani_out.to_jshtml())
