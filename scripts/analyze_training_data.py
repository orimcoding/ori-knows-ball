#!/usr/bin/env python3
"""
Analyze the quality of the training data and identify issues
"""
import pandas as pd
import numpy as np

def analyze_training_data():
    print("Loading training data...")
    df = pd.read_csv('data/training/master_training.csv')
    
    print('\n=== DATA QUALITY REPORT ===')
    print(f'Total matches: {len(df):,}')
    print(f'Total columns: {len(df.columns)}')
    
    print('\n=== Season Distribution ===')
    season_counts = df['season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f'{season}: {count:,} matches')
    
    print('\n=== Match Results ===')
    completed = df['result'].notna().sum()
    print(f'Matches with results: {completed:,} ({completed/len(df)*100:.1f}%)')
    if completed > 0:
        result_dist = df['result'].value_counts()
        print('Result distribution:')
        for result, count in result_dist.items():
            print(f'  {result}: {count:,} ({count/completed*100:.1f}%)')
    
    print('\n=== ELO Data Coverage ===')
    home_elo_missing = df['home_elo'].isnull().sum()
    away_elo_missing = df['away_elo'].isnull().sum()
    print(f'Home ELO missing: {home_elo_missing:,} ({home_elo_missing/len(df)*100:.1f}%)')
    print(f'Away ELO missing: {away_elo_missing:,} ({away_elo_missing/len(df)*100:.1f}%)')
    
    if home_elo_missing > 0:
        print('\nSample teams missing home ELO:')
        missing_home = df[df['home_elo'].isnull()][['season', 'date', 'home_team']].head(10)
        for _, row in missing_home.iterrows():
            print(f'  {row["season"]}: {row["home_team"]} on {row["date"]}')
    
    print('\n=== Team Statistics Coverage ===')
    # Count stats columns
    home_stats_cols = [col for col in df.columns if col.startswith('home_') and 
                      col not in ['home_team', 'home_goals', 'home_elo', 'home_season', 'home_win', 'home_xg']]
    away_stats_cols = [col for col in df.columns if col.startswith('away_') and 
                      col not in ['away_team', 'away_goals', 'away_elo', 'away_season', 'away_win', 'away_xg']]
    
    print(f'Home stats columns: {len(home_stats_cols)}')
    print(f'Away stats columns: {len(away_stats_cols)}')
    
    if home_stats_cols:
        home_stats_missing = df[home_stats_cols].isnull().sum()
        max_missing_home = home_stats_missing.max()
        print(f'Max missing home stats: {max_missing_home:,} ({max_missing_home/len(df)*100:.1f}%)')
        
        # Show which teams are missing stats
        if max_missing_home > 0:
            worst_col = home_stats_missing.idxmax()
            missing_teams = df[df[worst_col].isnull()][['season', 'home_team']].drop_duplicates().head(10)
            print(f'\nSample teams missing stats (column: {worst_col}):')
            for _, row in missing_teams.iterrows():
                print(f'  {row["season"]}: {row["home_team"]}')
    
    print('\n=== Team Name Standardization Issues ===')
    # Check for potential team name mismatches
    all_schedule_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    # Load ELO data to see what teams we have there
    try:
        elo_df = pd.read_csv('data/elo_ucl_teams/2024_ucl_elo.csv')
        elo_teams = set(elo_df['team'].str.strip().str.title().unique())
        
        # Find teams in schedule but not in ELO
        missing_in_elo = all_schedule_teams - elo_teams
        if missing_in_elo:
            print(f'\nTeams in schedule but missing from ELO data ({len(missing_in_elo)}):')
            for team in sorted(missing_in_elo)[:10]:  # Show first 10
                print(f'  {team}')
        
        # Find teams in ELO but not in schedule
        missing_in_schedule = elo_teams - all_schedule_teams
        if missing_in_schedule:
            print(f'\nTeams in ELO but missing from schedule ({len(missing_in_schedule)}):')
            for team in sorted(missing_in_schedule)[:10]:  # Show first 10
                print(f'  {team}')
                
    except Exception as e:
        print(f'Could not analyze ELO team names: {e}')
    
    print('\n=== Data Completeness for ML ===')
    # Count rows that have all essential data for ML
    essential_cols = ['home_goals', 'away_goals', 'home_elo', 'away_elo']
    complete_matches = df[essential_cols].notna().all(axis=1).sum()
    print(f'Matches with all essential data: {complete_matches:,} ({complete_matches/len(df)*100:.1f}%)')
    
    # Count matches with stats data
    if home_stats_cols:
        stats_complete = df[home_stats_cols + away_stats_cols].notna().all(axis=1).sum()
        print(f'Matches with complete team stats: {stats_complete:,} ({stats_complete/len(df)*100:.1f}%)')
    
    print('\n=== Feature Summary ===')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f'Numeric features: {len(numeric_cols)}')
    
    # Show some key stats
    if 'elo_diff' in df.columns:
        elo_diff_stats = df['elo_diff'].describe()
        print(f'\nELO difference stats:')
        print(f'  Mean: {elo_diff_stats["mean"]:.1f}')
        print(f'  Std: {elo_diff_stats["std"]:.1f}')
        print(f'  Range: [{elo_diff_stats["min"]:.1f}, {elo_diff_stats["max"]:.1f}]')

if __name__ == "__main__":
    analyze_training_data()
