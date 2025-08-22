from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FBREF_DIR = ROOT / "fbref_data"
ELO_DIR = ROOT / "data" / "elo_filtered"

def snake(c):
    return c.strip().lower().replace(" ", "_")

def list_seasons():
    return sorted(p.name for p in FBREF_DIR.iterdir() if p.is_dir() and p.name.isdigit())

def preview_columns(season, n=2):
    season_dir = FBREF_DIR / season
    tables = [
        "standard", "shooting", "passing", "passing_types", "possession",
        "playing_time", "defense", "keeper", "keeper_adv", "misc", "schedule"
    ]
    for t in tables:
        path = season_dir / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path, nrows=n)
            df.columns = [snake(c) for c in df.columns]
            print(f"\n[{season}] {t}.csv — {len(df.columns)} columns")
            print(list(df.columns))
        else:
            print(f"\n[{season}] {t}.csv — MISSING at {path}")

    elo_path = ELO_DIR / f"{season}_elo_filtered.csv"
    if elo_path.exists():
        df_elo = pd.read_csv(elo_path, nrows=n)
        df_elo.columns = [snake(c) for c in df_elo.columns]
        print(f"\nElo {elo_path.name} — {len(df_elo.columns)} columns")
        print(list(df_elo.columns))
    else:
        print(f"\nElo file missing: {elo_path}")

def load_clean_schedule(season):
    season = str(season)
    path = FBREF_DIR / season / "schedule.csv"
    df = pd.read_csv(path)
    df.columns = [snake(c) for c in df.columns]
    rename_map = {
        "home": "home_team", "home_team": "home_team",
        "away": "away_team", "away_team": "away_team",
        "xg": "home_xg", "xga": "away_xg",
        "date": "date", "round": "stage", "stage": "stage",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})
    if "score" in df.columns:
        s = df["score"].astype(str).str.lower()
        s = (
            s.str.replace(r"\(.*?\)", "", regex=True)
             .str.replace("after extra time", "", regex=False)
             .str.replace("aet", "", regex=False)
             .str.replace("pens", "", regex=False)
             .str.replace("penalties", "", regex=False)
             .str.strip()
        )
        s = s.str.replace("–", "-", regex=False)
        s = s.str.replace("—", "-", regex=False)
        parts = s.str.split("-", n=1, expand=True)
        df["home_goals"] = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
        df["away_goals"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")

    wanted = ["date", "stage", "home_team", "away_team",
              "home_goals", "away_goals", "home_xg", "away_xg"]
    for col in wanted:
        if col not in df.columns:
            df[col] = np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for g in ["home_goals", "away_goals"]:
        if df[g].dtype.name != "Int64":
            df[g] = pd.to_numeric(df[g], errors="coerce").astype("Int64")
    for x in ["home_xg", "away_xg"]:
        df[x] = pd.to_numeric(df[x], errors="coerce")

    df["season"] = int(season)
    df["goal_diff"] = df["home_goals"].astype("float") - df["away_goals"].astype("float")
    df["result"] = np.where(df["goal_diff"].gt(0), "H",
                   np.where(df["goal_diff"].lt(0), "A",
                   np.where(df["goal_diff"].eq(0), "D", None)))


    cols = ["season","date","stage","home_team","away_team",
            "home_goals","away_goals","home_xg","away_xg","goal_diff","result"]
    return df[cols].sort_values(["date","home_team","away_team"]).reset_index(drop=True)

if __name__ == "__main__":
    assert FBREF_DIR.exists(), f"Missing folder: {FBREF_DIR}"
    assert ELO_DIR.exists(), f"Missing folder: {ELO_DIR}"
    seasons = list_seasons()
    if not seasons:
        raise SystemExit("No season folders found under fbref_data/.")
    print("Seasons found:", seasons)
    sample = seasons[0]
    print("\n--- Previewing columns for season", sample, "---")
    preview_columns(sample)
    print("\n--- Cleaning schedule for season", sample, "---")
    sched = load_clean_schedule(sample)
    print(sched.head(8))
