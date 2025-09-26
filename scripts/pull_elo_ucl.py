import os
import pandas as pd
from soccerdata import ClubElo

# Example: paste UCL teams here for this season
ucl_teams = [
    "Frankfurt",       # Eintracht Frankfurt
    "Paris SG",        # PSG
    "Brugge",          # Club Brugge
    "Sporting",        # Sporting CP
    "St Gillis",       # Union Saint-Gilloise
    "Bayern",          # Bayern Munich
    "Arsenal",
    "Inter",           # Internazionale
    "Man City",        # Manchester City
    "Qarabag",         # Qarabag FK
    "Liverpool",
    "Barcelona",
    "Real Madrid",
    "Tottenham",
    "Dortmund",        # Borussia Dortmund
    "Juventus",
    "Leverkusen",      # Bayer Leverkusen
    "Bodo/Glimt",
    "FC København",    # Copenhagen
    "Slavia Praha",
    "Olympiakos",      # Olympiacos
    "Paphos",          # Pafos
    "Atlético",        # Atlético Madrid
    "Benfica",
    "Marseille",
    "Newcastle",
    "Villarreal",
    "Chelsea",
    "PSV",
    "Ajax",
    "Bilbao",          # Athletic Bilbao
    "Napoli",
    "Kairat Almaty",          # Kairat
    "Monaco",
    "Galatasaray",
    "Atalanta"
]


# Output folder
output_dir = "data/elo_ucl_teams"
os.makedirs(output_dir, exist_ok=True)

# Instantiate scraper
elo = ClubElo()

# Season year mapping (can adjust later)
season = 2025  # Example for current UCL
season_start = f"{season}-08-01"
season_end = f"{season+1}-06-30"

print(f"📊 Pulling Club Elo for {season}-{season+1} UCL teams...")

all_data = []

for team in ucl_teams:
    try:
        team_history = elo.read_team_history(team)
        # Filter to only the season range
        team_season = team_history.loc[season_start:season_end].copy()
        if not team_season.empty:
            # Add metadata
            team_season["season"] = f"{season}-{season+1}"
            all_data.append(team_season)
            print(f"✅ {team}: {len(team_season)} entries")
        else:
            print(f"⚠ No Elo data found for {team}")
    except Exception as e:
        print(f"❌ Failed for {team}: {e}")

# Combine & save
if all_data:
    df = pd.concat(all_data)
    output_path = os.path.join(output_dir, f"{season}_ucl_elo.csv")
    df.to_csv(output_path)
    print(f"💾 Saved Elo data for {len(ucl_teams)} teams → {output_path}")
else:
    print("❌ No data saved.")
