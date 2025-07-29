import soccerdata as sd
import os

competition = "UEFA-UCL"
stat_types = [
    "standard", "keeper", "keeper_adv", "shooting", "passing",
    "passing_types", "gca", "defense", "possession", "playing_time", "misc"
]

# ✅ PART 1: Scrape missing 2017–18 season
season = "2017"
print(f"\n📅 Scraping missing season {season}...")
try:
    fbref = sd.FBref(competition, season)
    os.makedirs(f"fbref_data/{season}", exist_ok=True)

    for stat in stat_types:
        try:
            df = fbref.read_team_season_stats(stat_type=stat)
            df.to_csv(f"fbref_data/{season}/{stat}.csv", index=False)
            print(f"✔ Saved {stat}.csv for {season}")
        except Exception as e:
            print(f"✘ Failed {stat} for {season}: {e}")
except Exception as e:
    print(f"❌ Could not initialize FBref for 2017: {e}")

# ✅ PART 2: Add schedule.csv to each season from 2017–2025
for season in map(str, range(2017, 2026)):
    try:
        print(f"\n📄 Adding schedule for {season}...")
        fbref = sd.FBref(competition, season)
        sched_df = fbref.read_schedule()
        os.makedirs(f"fbref_data/{season}", exist_ok=True)
        sched_df.to_csv(f"fbref_data/{season}/schedule.csv", index=False)
        print(f"✔ Saved schedule.csv for {season}")
    except Exception as e:
        print(f"✘ Failed to save schedule for {season}: {e}")
