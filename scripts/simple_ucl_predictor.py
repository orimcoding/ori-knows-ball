#!/usr/bin/env python3
"""
Simplified UCL Match Predictor focused on ELO and basic features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleUCLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def standardize_team_name(self, name):
        """Standardize team names"""
        if pd.isna(name):
            return name
        
        name = str(name).strip()
        
        team_mapping = {
            'Man City': 'Manchester City', 'PSG': 'Paris Saint-Germain',
            'Paris S-G': 'Paris Saint-Germain', 'Psg': 'Paris Saint-Germain',
            'Bayern': 'Bayern Munich', 'Dortmund': 'Borussia Dortmund',
            'Inter': 'Inter Milan', 'Atletico': 'Atlético Madrid',
            'Atletico Madrid': 'Atlético Madrid', 'Ajax': 'Ajax Amsterdam',
            'PSV': 'PSV Eindhoven', 'Porto': 'FC Porto',
            'Sporting CP': 'Sporting Lisbon', 'Celtic': 'Celtic FC',
            'RB Leipzig': 'RB Leipzig', 'Rb Leipzig': 'RB Leipzig',
            'Salzburg': 'RB Salzburg', 'Rb Salzburg': 'RB Salzburg',
            'Newcastle': 'Newcastle United', 'Tottenham': 'Tottenham Hotspur',
            'Eint Frankfurt': 'Eintracht Frankfurt', 'Frankfurt': 'Eintracht Frankfurt',
            'FC Copenhagen': 'FC Kobenhavn', 'Fc Copenhagen': 'FC Kobenhavn',
            'Newcastle Utd': 'Newcastle United', 'Union SG': 'Union Saint-Gilloise',
            'Qaırat Almaty': 'Qairat Almaty', 'Pafos FC': 'Pafos FC'
        }
        
        return team_mapping.get(name, name)
        
    def prepare_training_data(self):
        """Load and prepare training data with core features only"""
        print("Loading training data...")
        
        try:
            df = pd.read_csv('data/training/clean_training.csv')
            print(f"✅ Loaded {len(df):,} clean matches")
        except FileNotFoundError:
            print("❌ Clean training data not found. Run build_training_data.py first.")
            return None, None
            
        # Core features that we can reliably get
        core_features = [
            'home_elo', 'away_elo', 'home_xg', 'away_xg'
        ]
        
        # Check which features are available
        available_features = [f for f in core_features if f in df.columns]
        print(f"Available core features: {available_features}")
        
        if len(available_features) < 2:
            print("❌ Not enough core features available")
            return None, None
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Engineer additional features
        if 'home_elo' in X.columns and 'away_elo' in X.columns:
            X['elo_diff'] = X['home_elo'] - X['away_elo']
            X['elo_sum'] = X['home_elo'] + X['away_elo']
            X['elo_ratio'] = X['home_elo'] / (X['away_elo'] + 1)
            
        if 'home_xg' in X.columns and 'away_xg' in X.columns:
            X['xg_diff'] = X['home_xg'] - X['away_xg']
            X['xg_total'] = X['home_xg'] + X['away_xg']
        
        # Add home advantage (historical average)
        X['home_advantage'] = 0.15  # 15% home advantage based on our data
        
        # Target variable
        y = df['result'].copy()
        
        # Remove rows with missing values
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Final dataset: {len(X):,} matches, {len(X.columns)} features")
        print(f"Features: {list(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train_model(self, X, y):
        """Train the prediction model"""
        print("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train ensemble model
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train individual models
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        lr.fit(X_train_scaled, y_train)
        
        # Create ensemble
        self.model = VotingClassifier([
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ], voting='soft')
        
        # Train ensemble
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained with {accuracy:.1%} accuracy")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (from Random Forest)
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Feature Importance:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']:<15} {row['importance']:.3f}")
        
        self.is_trained = True
        return accuracy
    
    def predict_upcoming_matches(self):
        """Predict upcoming UCL matches"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
            
        print("\n=== PREDICTING UPCOMING MATCHES ===")
        
        # Load 2025 schedule
        try:
            schedule = pd.read_csv('fbref_data/2025/schedule.csv')
        except FileNotFoundError:
            print("❌ 2025 schedule not found")
            return None
        
        # Filter upcoming matches (no score yet)
        upcoming = schedule[schedule['score'].isna()].copy()
        
        if len(upcoming) == 0:
            print("No upcoming matches found")
            return None
            
        print(f"Found {len(upcoming)} upcoming matches")
        
        # Standardize team names
        upcoming['home_team'] = upcoming['home_team'].apply(self.standardize_team_name)
        upcoming['away_team'] = upcoming['away_team'].apply(self.standardize_team_name)
        
        # Load ELO ratings
        try:
            elo_data = pd.read_csv('data/elo_ucl_teams/2024_ucl_elo.csv')
            elo_data['team'] = elo_data['team'].apply(self.standardize_team_name)
            
            # Get latest ELO for each team
            latest_elo = elo_data.groupby('team')['elo'].last().reset_index()
            elo_dict = dict(zip(latest_elo['team'], latest_elo['elo']))
            
        except FileNotFoundError:
            print("❌ ELO data not found")
            return None
            
        # Add ELO ratings
        upcoming['home_elo'] = upcoming['home_team'].map(elo_dict)
        upcoming['away_elo'] = upcoming['away_team'].map(elo_dict)
        
        # Add default xG values (will be more accurate as season progresses)
        upcoming['home_xg'] = 1.5  # Average home xG
        upcoming['away_xg'] = 1.2  # Average away xG
        
        # Create feature matrix (same as training)
        X_pred = pd.DataFrame()
        
        if 'home_elo' in self.feature_names:
            X_pred['home_elo'] = upcoming['home_elo']
        if 'away_elo' in self.feature_names:
            X_pred['away_elo'] = upcoming['away_elo']
        if 'home_xg' in self.feature_names:
            X_pred['home_xg'] = upcoming['home_xg']
        if 'away_xg' in self.feature_names:
            X_pred['away_xg'] = upcoming['away_xg']
            
        # Engineer same features as training
        if 'elo_diff' in self.feature_names:
            X_pred['elo_diff'] = X_pred['home_elo'] - X_pred['away_elo']
        if 'elo_sum' in self.feature_names:
            X_pred['elo_sum'] = X_pred['home_elo'] + X_pred['away_elo']
        if 'elo_ratio' in self.feature_names:
            X_pred['elo_ratio'] = X_pred['home_elo'] / (X_pred['away_elo'] + 1)
        if 'xg_diff' in self.feature_names:
            X_pred['xg_diff'] = X_pred['home_xg'] - X_pred['away_xg']
        if 'xg_total' in self.feature_names:
            X_pred['xg_total'] = X_pred['home_xg'] + X_pred['away_xg']
        if 'home_advantage' in self.feature_names:
            X_pred['home_advantage'] = 0.15
            
        # Remove matches with missing ELO data
        valid_mask = X_pred.notna().all(axis=1)
        X_pred = X_pred[valid_mask]
        upcoming_valid = upcoming[valid_mask].reset_index(drop=True)
        
        if len(X_pred) == 0:
            print("❌ No matches with complete data for prediction")
            return None
            
        print(f"Predicting {len(X_pred)} matches with complete data")
        
        # Make predictions
        predictions = self.model.predict(X_pred)
        probabilities = self.model.predict_proba(X_pred)
        
        # Create results dataframe
        results = upcoming_valid[['round', 'date', 'time', 'home_team', 'away_team', 'venue']].copy()
        results['predicted_result'] = predictions
        
        # Get class labels
        classes = self.model.classes_
        for i, cls in enumerate(classes):
            if cls == 'H':
                results['home_win_prob'] = probabilities[:, i]
            elif cls == 'A':
                results['away_win_prob'] = probabilities[:, i]
            elif cls == 'D':
                results['draw_prob'] = probabilities[:, i]
                
        results['confidence'] = np.max(probabilities, axis=1)
        
        # Sort by date
        results['date'] = pd.to_datetime(results['date'])
        results = results.sort_values('date')
        
        return results
    
    def display_predictions(self, predictions):
        """Display predictions nicely"""
        if predictions is None or len(predictions) == 0:
            return
            
        print(f"\n🔮 UPCOMING UCL MATCH PREDICTIONS")
        print("="*80)
        
        # Group by round for better organization
        for round_name in predictions['round'].unique()[:3]:  # Show next 3 rounds
            round_matches = predictions[predictions['round'] == round_name].head(10)
            
            print(f"\n📋 {round_name}")
            print("-" * 60)
            
            for _, match in round_matches.iterrows():
                date_str = match['date'].strftime('%Y-%m-%d') if pd.notna(match['date']) else 'TBD'
                time_str = match['time'] if pd.notna(match['time']) else 'TBD'
                
                home_prob = match.get('home_win_prob', 0)
                away_prob = match.get('away_win_prob', 0)
                draw_prob = match.get('draw_prob', 0)
                
                # Determine prediction emoji
                pred_emoji = "🏠" if match['predicted_result'] == 'H' else "🛫" if match['predicted_result'] == 'A' else "🤝"
                
                print(f"\n📅 {date_str} {time_str}")
                print(f"{pred_emoji} {match['home_team']} vs {match['away_team']}")
                print(f"🎯 Prediction: {match['predicted_result']} (Confidence: {match['confidence']:.1%})")
                print(f"📊 H {home_prob:.1%} | D {draw_prob:.1%} | A {away_prob:.1%}")
        
        print("\n" + "="*80)
        
        # Summary
        high_confidence = predictions[predictions['confidence'] > 0.6]
        print(f"📈 High confidence predictions (>60%): {len(high_confidence)}/{len(predictions)}")
        
        pred_counts = predictions['predicted_result'].value_counts()
        print(f"🏆 Predicted outcomes: H: {pred_counts.get('H', 0)}, D: {pred_counts.get('D', 0)}, A: {pred_counts.get('A', 0)}")


def main():
    print("🚀 Starting UCL Super Predictor...")
    
    predictor = SimpleUCLPredictor()
    
    # Load and prepare data
    X, y = predictor.prepare_training_data()
    if X is None:
        return
    
    # Train model
    accuracy = predictor.train_model(X, y)
    
    # Predict upcoming matches
    predictions = predictor.predict_upcoming_matches()
    predictor.display_predictions(predictions)
    
    # Save predictions
    if predictions is not None and len(predictions) > 0:
        predictions.to_csv('data/predictions/ucl_predictions.csv', index=False)
        print(f"\n💾 Saved {len(predictions)} predictions to data/predictions/ucl_predictions.csv")
        
        # Save a summary of the most confident predictions
        confident_preds = predictions[predictions['confidence'] > 0.6].copy()
        if len(confident_preds) > 0:
            confident_preds.to_csv('data/predictions/confident_predictions.csv', index=False)
            print(f"💎 Saved {len(confident_preds)} high-confidence predictions")
    
    print(f"\n✅ UCL Predictor completed with {accuracy:.1%} accuracy!")

if __name__ == "__main__":
    main()
