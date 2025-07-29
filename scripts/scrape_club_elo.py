import soccerdata as sd
import pandas as pd

# Create ClubElo scraper
elo = sd.ClubElo()

# Define top teams to pull Elo for
top_teams = [
    "Barcelona", "Real Madrid", "Manchester City", "Bayern Munich",
    "PSG", "Liverpool", "Chelsea", "Atletico Madrid", "Juventus"
]

# Read Elo data
elo_df = elo.read_by_club(top_teams)

# Save to CSV
elo_df.to_csv("club_elo_top_teams.csv", index=False)

print("✔ Saved club Elo ratings")
