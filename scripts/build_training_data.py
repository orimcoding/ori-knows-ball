# scripts/build_training_data.py
import os
import glob
import pandas as pd
import numpy as np

# =======================================================
# 1. Load all schedules (fbref_data/{season}/schedule.csv)
# =======================================================
# Look for schedules inside folders OR directly in fbref_data
schedule_files = glob.glob("fbref_data/*/schedule.csv")
if not schedule_files:
    schedule_files = glob.glob("fbref_data/*schedule.csv")

print("Found schedule files:", schedule_files)

schedules = []
for f in schedule_files:
    season = os.path.basename(os.path.dirname(f))  # folder name like "2025"
    df = pd.read_csv(f)
    df["season"] = season
    schedules.append(df)

sched = pd.concat(schedules, ignore_index=True)

# Parse score → goals
sched["home_goals"] = sched["score"].str.extract(r"(\d+)–").astype(float)
sched["away_goals"] = sched["score"].str.extract(r"–(\d+)").astype(float)

# Define result labels
def result(row):
    if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
        return None
    if row["home_goals"] > row["away_goals"]:
        return "H"
    elif row["home_goals"] < row["away_goals"]:
        return "A"
    else:
        return "D"

sched["result"] = sched.apply(result, axis=1)

# =======================================================
# Team Name Standardization
# =======================================================
def standardize_team_name(name):
    """Standardize team names across all data sources"""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    
    # Create comprehensive mapping
    team_mapping = {
        # Premier League
        'Man City': 'Manchester City',
        'Man United': 'Manchester United',
        'Man Utd': 'Manchester United',
        'Newcastle': 'Newcastle United',
        'Tottenham': 'Tottenham Hotspur',
        'Leicester': 'Leicester City',
        'West Ham': 'West Ham United',
        'Brighton': 'Brighton & Hove Albion',
        
        # La Liga
        'Atletico': 'Atlético Madrid',
        'Atletico Madrid': 'Atlético Madrid',
        'Atlético Madrid': 'Atlético Madrid',
        'Athletic Club': 'Athletic Bilbao',
        'Athletic Bilbao': 'Athletic Bilbao',
        'Bilbao': 'Athletic Bilbao',
        'Real Sociedad': 'Real Sociedad',
        'Celta Vigo': 'Celta de Vigo',
        
        # Bundesliga
        'Bayern': 'Bayern Munich',
        'Bayern München': 'Bayern Munich',
        'Dortmund': 'Borussia Dortmund',
        'RB Leipzig': 'RB Leipzig',
        'Rb Leipzig': 'RB Leipzig',
        'Leverkusen': 'Bayer Leverkusen',
        'Eint Frankfurt': 'Eintracht Frankfurt',
        'Frankfurt': 'Eintracht Frankfurt',
        'Wolfsburg': 'VfL Wolfsburg',
        'Hoffenheim': 'TSG Hoffenheim',
        
        # Serie A
        'Inter': 'Inter Milan',
        'Inter Milan': 'Inter Milan',
        'AC Milan': 'Milan',
        'Ac Milan': 'Milan',
        'Juventus': 'Juventus',
        'Roma': 'AS Roma',
        'As Roma': 'AS Roma',
        'Lazio': 'SS Lazio',
        'Napoli': 'Napoli',
        'Atalanta': 'Atalanta',
        'Fiorentina': 'Fiorentina',
        
        # Ligue 1
        'PSG': 'Paris Saint-Germain',
        'Paris S-G': 'Paris Saint-Germain',
        'Paris SG': 'Paris Saint-Germain',
        'Psg': 'Paris Saint-Germain',
        'Monaco': 'AS Monaco',
        'As Monaco': 'AS Monaco',
        'Marseille': 'Olympique Marseille',
        'Lyon': 'Olympique Lyon',
        'Lille': 'Lille OSC',
        
        # Other European teams
        'Ajax': 'Ajax Amsterdam',
        'PSV': 'PSV Eindhoven',
        'Feyenoord': 'Feyenoord Rotterdam',
        'Anderlecht': 'RSC Anderlecht',
        'Club Brugge': 'Club Brugge KV',
        'Brugge': 'Club Brugge KV',
        'Porto': 'FC Porto',
        'Benfica': 'SL Benfica',
        'Sporting CP': 'Sporting Lisbon',
        'Celtic': 'Celtic FC',
        'Rangers': 'Rangers FC',
        'Salzburg': 'RB Salzburg',
        'Rb Salzburg': 'RB Salzburg',
        'Basel': 'FC Basel',
        'Young Boys': 'BSC Young Boys',
        'Shakhtar': 'Shakhtar Donetsk',
        'Dynamo Kyiv': 'Dynamo Kiev',
        'Dinamo Zagreb': 'Dinamo Zagreb',
        'Red Star': 'Red Star Belgrade',
        'Galatasaray': 'Galatasaray SK',
        'Fenerbahçe': 'Fenerbahce SK',
        'Beşiktaş': 'Besiktas JK',
        'Başakşehir': 'Istanbul Basaksehir',
        
        # Eastern European
        'Viktoria Plzeň': 'Viktoria Plzen',
        'Slavia Prague': 'Slavia Praha',
        'Sparta Prague': 'Sparta Praha',
        
        # Scandinavian
        'FC Copenhagen': 'FC Kobenhavn',
        'Fc Copenhagen': 'FC Kobenhavn',
        'Malmö FF': 'Malmo FF',
        
        # Other
        'APOEL': 'APOEL FC',
        'Apoel Fc': 'APOEL FC',
        'AEK Athens': 'AEK Athens FC',
        'Aek Athens': 'AEK Athens FC',
        'Olympiakos': 'Olympiakos FC',
        'Panathinaikos': 'Panathinaikos FC',
    }
    
    # First try exact match
    if name in team_mapping:
        return team_mapping[name]
    
    # Try case-insensitive match
    for key, value in team_mapping.items():
        if name.lower() == key.lower():
            return value
    
    # Return original name if no mapping found
    return name

# Standardize team names
sched["home_team"] = sched["home_team"].apply(standardize_team_name)
sched["away_team"] = sched["away_team"].apply(standardize_team_name) 
sched["date"] = pd.to_datetime(sched["date"], errors="coerce")

# =======================================================
# 2. Load Elo (from both sources)
# =======================================================
# Load from elo_ucl_teams (detailed time series)
elo_files = glob.glob("data/elo_ucl_teams/*_ucl_elo.csv")
elos = []
for f in elo_files:
    season = os.path.basename(f).split("_")[0]  # "2024"
    df = pd.read_csv(f)
    df["season"] = season
    df["from"] = pd.to_datetime(df["from"], errors="coerce")
    elos.append(df)

# Load from elo_filtered (seasonal averages)
elo_filtered_files = glob.glob("data/elo_filtered/*_elo_filtered.csv")
for f in elo_filtered_files:
    season = os.path.basename(f).split("_")[0]  # "2024"
    df = pd.read_csv(f)
    df["season"] = season
    # Create a synthetic date for season start (August 1st)
    df["from"] = pd.to_datetime(f"{season}-08-01")
    elos.append(df)

if elos:
    elo = pd.concat(elos, ignore_index=True)
    elo["team"] = elo["team"].str.strip().apply(standardize_team_name)
else:
    print("Warning: No ELO data found!")
    elo = pd.DataFrame()

# Helper: get last Elo before match date
def get_latest_elo(team, date, season):
    df = elo[(elo["team"] == team) & (elo["season"] == season) & (elo["from"] <= date)]
    if not df.empty:
        return df.sort_values("from").iloc[-1]["elo"]
    return np.nan

sched["home_elo"] = sched.apply(lambda r: get_latest_elo(r["home_team"], r["date"], r["season"]), axis=1)
sched["away_elo"] = sched.apply(lambda r: get_latest_elo(r["away_team"], r["date"], r["season"]), axis=1)
sched["elo_diff"] = sched["home_elo"] - sched["away_elo"]

# =======================================================
# 3. Load FBref Stats
# =======================================================
stats_categories = [
    "defense", "keeper_adv", "keeper", "misc",
    "passing_types", "passing", "playing_time",
    "possession", "shooting", "standard"
]

fbref_data = []
for season_folder in glob.glob("fbref_data/*/"):
    season = os.path.basename(os.path.normpath(season_folder))
    season_stats = []
    for cat in stats_categories:
        f = os.path.join(season_folder, f"{cat}.csv")
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                df["season"] = season
                df["stat_type"] = cat
                # Extract team name from URL
                if "url" in df.columns:
                    df["team"] = df["url"].str.extract(r"/([^/]+)-Stats$")[0].str.replace("-", " ").str.strip()
                    df["team"] = df["team"].apply(standardize_team_name)
                season_stats.append(df)
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
    if season_stats:
        fbref_data.append(pd.concat(season_stats, ignore_index=True))

fbref = pd.concat(fbref_data, ignore_index=True)

# Aggregate stats by team/season
team_stats = fbref.groupby(["season", "team"]).mean(numeric_only=True).reset_index()

# =======================================================
# 4. Merge schedule + stats
# =======================================================
sched = sched.merge(team_stats.add_prefix("home_"), left_on=["season", "home_team"], right_on=["home_season", "home_team"], how="left")
sched = sched.merge(team_stats.add_prefix("away_"), left_on=["season", "away_team"], right_on=["away_season", "away_team"], how="left")

# =======================================================
# 5. Define target variable
# =======================================================
sched["home_win"] = (sched["home_goals"] > sched["away_goals"]).astype(float)
sched["away_win"] = (sched["away_goals"] > sched["home_goals"]).astype(float)
sched["draw"] = (sched["home_goals"] == sched["away_goals"]).astype(float)

# =======================================================
# 6. Data Quality Check & Final Processing
# =======================================================
print("\n=== DATA QUALITY SUMMARY ===")
print(f"Total matches: {len(sched):,}")
print(f"Matches with results: {sched['result'].notna().sum():,} ({sched['result'].notna().sum()/len(sched)*100:.1f}%)")
print(f"Home ELO coverage: {sched['home_elo'].notna().sum():,} ({sched['home_elo'].notna().sum()/len(sched)*100:.1f}%)")
print(f"Away ELO coverage: {sched['away_elo'].notna().sum():,} ({sched['away_elo'].notna().sum()/len(sched)*100:.1f}%)")

# Count how many matches have complete data for ML
essential_cols = ['home_goals', 'away_goals', 'home_elo', 'away_elo']
complete_for_ml = sched[essential_cols].notna().all(axis=1).sum()
print(f"Matches ready for ML: {complete_for_ml:,} ({complete_for_ml/len(sched)*100:.1f}%)")

# Show season breakdown
print(f"\nSeason distribution:")
season_counts = sched['season'].value_counts().sort_index()
for season, count in season_counts.items():
    season_complete = sched[sched['season'] == season][essential_cols].notna().all(axis=1).sum()
    print(f"  {season}: {count:,} total, {season_complete:,} ML-ready ({season_complete/count*100:.1f}%)")

# =======================================================
# 7. Save Final Dataset
# =======================================================
os.makedirs("data/training", exist_ok=True)
out_path = "data/training/master_training.csv"
sched.to_csv(out_path, index=False)
print(f"\n✅ Saved master dataset → {out_path}")
print(f"Shape: {sched.shape}")

# Also save a clean version with only ML-ready matches
if complete_for_ml > 0:
    clean_data = sched[sched[essential_cols].notna().all(axis=1)].copy()
    clean_path = "data/training/clean_training.csv"
    clean_data.to_csv(clean_path, index=False)
    print(f"✅ Saved clean dataset → {clean_path}")
    print(f"Clean shape: {clean_data.shape}")
    
    # Show target variable distribution in clean data
    if 'result' in clean_data.columns:
        result_dist = clean_data['result'].value_counts()
        print(f"\nClean data result distribution:")
        total_with_results = result_dist.sum()
        for result, count in result_dist.items():
            print(f"  {result}: {count:,} ({count/total_with_results*100:.1f}%)")
else:
    print("⚠️  No matches have complete data for ML!")
