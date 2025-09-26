#!/usr/bin/env python3
"""
Quick analysis of the clean training dataset
"""
import pandas as pd
import numpy as np

def analyze_clean_data():
    try:
        df = pd.read_csv('data/training/clean_training.csv')
        
        print("=== CLEAN TRAINING DATA ANALYSIS ===")
        print(f"Total matches: {len(df):,}")
        print(f"Features: {len(df.columns)}")
        
        # Season distribution
        print(f"\n=== Season Breakdown ===")
        season_dist = df['season'].value_counts().sort_index()
        for season, count in season_dist.items():
            print(f"{season}: {count:,} matches")
        
        # Result distribution
        print(f"\n=== Result Distribution ===")
        result_dist = df['result'].value_counts()
        total = result_dist.sum()
        for result, count in result_dist.items():
            print(f"{result} (Home/Away/Draw): {count:,} ({count/total*100:.1f}%)")
        
        # ELO statistics
        print(f"\n=== ELO Statistics ===")
        print(f"Home ELO: {df['home_elo'].mean():.1f} ± {df['home_elo'].std():.1f}")
        print(f"Away ELO: {df['away_elo'].mean():.1f} ± {df['away_elo'].std():.1f}")
        print(f"ELO Difference: {df['elo_diff'].mean():.1f} ± {df['elo_diff'].std():.1f}")
        
        # Feature completeness
        print(f"\n=== Feature Completeness ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_stats = df[numeric_cols].isnull().sum()
        
        print(f"Numeric features: {len(numeric_cols)}")
        if missing_stats.sum() > 0:
            print(f"Features with missing data: {(missing_stats > 0).sum()}")
            print(f"Max missing values: {missing_stats.max()}")
        else:
            print("✅ No missing numeric data!")
        
        # Team coverage
        print(f"\n=== Team Coverage ===")
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        print(f"Unique teams: {len(all_teams)}")
        
        # Most frequent teams
        home_counts = df['home_team'].value_counts()
        away_counts = df['away_team'].value_counts()
        total_counts = home_counts.add(away_counts, fill_value=0).sort_values(ascending=False)
        
        print(f"\nMost frequent teams (top 10):")
        for team, count in total_counts.head(10).items():
            print(f"  {team}: {int(count)} matches")
        
        # Quick ML readiness check
        print(f"\n=== ML Readiness ===")
        ml_features = ['home_elo', 'away_elo', 'elo_diff']
        
        # Add team stats if available
        team_stat_cols = [col for col in df.columns if 
                         (col.startswith('home_') or col.startswith('away_')) and 
                         col not in ['home_team', 'away_team', 'home_goals', 'away_goals', 
                                   'home_win', 'away_win', 'home_season', 'away_season']]
        
        if team_stat_cols:
            print(f"Available team stat features: {len(team_stat_cols)}")
            ml_features.extend(team_stat_cols[:10])  # Sample first 10
        
        complete_rows = df[ml_features].notna().all(axis=1).sum()
        print(f"Rows with complete ML features: {complete_rows:,} ({complete_rows/len(df)*100:.1f}%)")
        
        # Target variable balance
        print(f"\n=== Target Variable Balance ===")
        home_win_rate = df['home_win'].mean()
        away_win_rate = df['away_win'].mean()
        draw_rate = df['draw'].mean()
        
        print(f"Home win rate: {home_win_rate:.3f}")
        print(f"Away win rate: {away_win_rate:.3f}")
        print(f"Draw rate: {draw_rate:.3f}")
        print(f"Home advantage: {home_win_rate - away_win_rate:.3f}")
        
        print(f"\n✅ Clean dataset is ready for machine learning!")
        
    except FileNotFoundError:
        print("❌ Clean training data not found. Run build_training_data.py first.")
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

if __name__ == "__main__":
    analyze_clean_data()
