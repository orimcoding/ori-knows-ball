import os
import pandas as pd
from soccerdata import ClubElo

# Create output directory
output_dir = "data/elo_filtered"
os.makedirs(output_dir, exist_ok=True)

# Define season ranges (season end = June 30)
seasons = {
    2017: "2018-06-30",
    2018: "2019-06-30",
    2019: "2020-06-30",
    2020: "2021-06-30",
    2021: "2022-06-30",
    2022: "2023-06-30",
    2023: "2024-06-30",
    2024: "2025-06-30",
    2025: "2026-06-30"
}

# Instantiate ClubElo
elo = ClubElo()

# Loop through seasons
for season, date_str in seasons.items():
    try:
        print(f"📈 Pulling Club Elo for season ending {date_str}...")
        df = elo.read_by_date(date_str)
        
        # Filter Elo > 1650
        filtered = df[df["elo"] > 1650].copy()
        filtered.sort_values("elo", ascending=False, inplace=True)

        # Save CSV
        output_path = os.path.join(output_dir, f"{season}_elo_filtered.csv")
        filtered.to_csv(output_path)
        print(f"✅ Saved {len(filtered)} teams for season {season} to {output_path}")
    except Exception as e:
        print(f"❌ Failed for season {season}: {e}")
