import soccerdata as sd
import os

# FBref stat types we want to scrape
stat_types = [
    "standard", "keeper", "keeper_adv", "shooting", "passing",
    "passing_types", "gca", "defense", "possession", "playing_time", "misc"
]

# Champions League seasons: 2017–18 to 2024–25
seasons = range(2018, 2026)
competition = "UEFA-UCL"

for season in seasons:
    print(f"Scraping season {season}...")
    fbref = sd.FBref(competition, str(season))
    os.makedirs(f"fbref_data/{season}", exist_ok=True)

    for stat in stat_types:
        try:
            df = fbref.read_team_season_stats(stat_type=stat)
            df.to_csv(f"fbref_data/{season}/{stat}.csv", index=False)
            print(f"✔ Saved {stat} for {season}")
        except Exception as e:
            print(f"✘ Failed {stat} for {season}: {e}")

    # Save match schedule (once per season)
    try:
        sched_df = fbref.read_schedule()
        sched_df.to_csv(f"fbref_data/{season}/schedule.csv", index=False)
        print(f"✔ Saved match schedule for {season}")
    except Exception as e:
        print(f"✘ Failed to get schedule for {season}: {e}")
