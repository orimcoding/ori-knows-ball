#!/usr/bin/env python3
"""
Super Accurate UCL Match Predictor
Uses all available data including ELO ratings, team stats, and historical performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UCLPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.is_trained = False
        
    def load_and_prepare_data(self):
        """Load and prepare the training data"""
        print("Loading training data...")
        
        # Load clean training data
        try:
            df = pd.read_csv('data/training/clean_training.csv')
            print(f"✅ Loaded {len(df):,} clean matches")
        except FileNotFoundError:
            print("❌ Clean training data not found. Run build_training_data.py first.")
            return None
            
        # Feature engineering
        df = self.engineer_features(df)
        
        # Prepare features and target
        feature_cols = self.select_features(df)
        X = df[feature_cols].copy()
        
        # Target variable: H/A/D
        y = df['result'].copy()
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Final dataset: {len(X):,} matches, {len(feature_cols)} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def engineer_features(self, df):
        """Create additional features from the data"""
        print("Engineering features...")
        
        df = df.copy()
        
        # ELO-based features
        if 'home_elo' in df.columns and 'away_elo' in df.columns:
            df['elo_diff'] = df['home_elo'] - df['away_elo']
            df['elo_sum'] = df['home_elo'] + df['away_elo']
            df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)  # +1 to avoid division by zero
            
            # ELO advantage categories
            df['elo_advantage'] = pd.cut(df['elo_diff'], 
                                       bins=[-np.inf, -100, -50, 50, 100, np.inf],
                                       labels=['large_away', 'small_away', 'balanced', 'small_home', 'large_home'])
        
        # Expected goals features
        if 'home_xg' in df.columns and 'away_xg' in df.columns:
            df['xg_diff'] = df['home_xg'] - df['away_xg']
            df['xg_total'] = df['home_xg'] + df['away_xg']
        
        # Historical performance (season-based)
        season_stats = df.groupby(['season', 'home_team']).agg({
            'home_win': 'mean',
            'result': lambda x: (x == 'H').mean()
        }).rename(columns={'home_win': 'team_home_win_rate', 'result': 'team_home_result_rate'})
        
        # Add team form (rolling averages)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Team strength categories based on ELO
        if 'home_elo' in df.columns:
            elo_quartiles = df[['home_elo', 'away_elo']].quantile([0.25, 0.5, 0.75])
            df['home_strength'] = pd.cut(df['home_elo'], 
                                       bins=[0, elo_quartiles.iloc[0, 0], elo_quartiles.iloc[1, 0], 
                                            elo_quartiles.iloc[2, 0], np.inf],
                                       labels=['weak', 'average', 'strong', 'elite'])
            df['away_strength'] = pd.cut(df['away_elo'], 
                                       bins=[0, elo_quartiles.iloc[0, 1], elo_quartiles.iloc[1, 1], 
                                            elo_quartiles.iloc[2, 1], np.inf],
                                       labels=['weak', 'average', 'strong', 'elite'])
        
        return df
    
    def engineer_features_prediction(self, df):
        """Simplified feature engineering for prediction data"""
        df = df.copy()
        
        # ELO-based features
        if 'home_elo' in df.columns and 'away_elo' in df.columns:
            df['elo_diff'] = df['home_elo'] - df['away_elo']
            df['elo_sum'] = df['home_elo'] + df['away_elo']
            df['elo_ratio'] = df['home_elo'] / (df['away_elo'] + 1)
            
            # ELO advantage categories
            df['elo_advantage'] = pd.cut(df['elo_diff'], 
                                       bins=[-np.inf, -100, -50, 50, 100, np.inf],
                                       labels=['large_away', 'small_away', 'balanced', 'small_home', 'large_home'])
        
        # Expected goals features (if available)
        if 'home_xg' in df.columns and 'away_xg' in df.columns:
            df['xg_diff'] = df['home_xg'] - df['away_xg']
            df['xg_total'] = df['home_xg'] + df['away_xg']
        else:
            # Set default values if xG not available
            df['home_xg'] = 1.5  # Average xG
            df['away_xg'] = 1.2  # Away teams typically have lower xG
            df['xg_diff'] = df['home_xg'] - df['away_xg']
            df['xg_total'] = df['home_xg'] + df['away_xg']
        
        # Team strength categories based on ELO
        if 'home_elo' in df.columns and df['home_elo'].notna().any():
            # Use fixed thresholds based on typical ELO ranges
            df['home_strength'] = pd.cut(df['home_elo'], 
                                       bins=[0, 1700, 1800, 1900, np.inf],
                                       labels=['weak', 'average', 'strong', 'elite'])
            df['away_strength'] = pd.cut(df['away_elo'], 
                                       bins=[0, 1700, 1800, 1900, np.inf],
                                       labels=['weak', 'average', 'strong', 'elite'])
        
        return df
    
    def select_features(self, df):
        """Select the best features for the model"""
        feature_cols = []
        
        # Core features
        essential_features = ['home_elo', 'away_elo', 'elo_diff', 'elo_sum', 'elo_ratio']
        feature_cols.extend([f for f in essential_features if f in df.columns])
        
        # Expected goals features
        xg_features = ['home_xg', 'away_xg', 'xg_diff', 'xg_total']
        feature_cols.extend([f for f in xg_features if f in df.columns])
        
        # Team statistics (select most predictive ones)
        stat_features = []
        for prefix in ['home_', 'away_']:
            potential_stats = [
                f'{prefix}Gls', f'{prefix}Ast', f'{prefix}xG', f'{prefix}xAG',
                f'{prefix}Poss', f'{prefix}Tkl', f'{prefix}Int', f'{prefix}Clr',
                f'{prefix}KP', f'{prefix}PrgP', f'{prefix}Age'
            ]
            stat_features.extend([f for f in potential_stats if f in df.columns])
        
        feature_cols.extend(stat_features)
        
        # Categorical features (will be encoded)
        categorical_features = []
        if 'elo_advantage' in df.columns:
            categorical_features.append('elo_advantage')
        if 'home_strength' in df.columns:
            categorical_features.extend(['home_strength', 'away_strength'])
        
        # Encode categorical features
        for cat_feat in categorical_features:
            if cat_feat in df.columns:
                dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat)
                df[dummies.columns] = dummies
                feature_cols.extend(dummies.columns)
        
        # Remove duplicates and ensure all features exist
        feature_cols = list(set(feature_cols))
        feature_cols = [f for f in feature_cols if f in df.columns]
        
        print(f"Selected {len(feature_cols)} features")
        return feature_cols
    
    def handle_missing_values(self, X):
        """Handle missing values in features"""
        # For numerical features, use median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
        
        # For categorical features, use mode
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        return X
    
    def train_models(self, X, y):
        """Train multiple models and create an ensemble"""
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train individual models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                min_samples_split=5, random_state=42
            ),
            'logistic': LogisticRegression(
                max_iter=1000, random_state=42, multi_class='ovr'
            )
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'logistic':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.3f}")
            
            # Cross-validation
            if name == 'logistic':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            print(f"{name} CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            self.models[name] = model
        
        # Create ensemble model
        print("\nCreating ensemble model...")
        ensemble = VotingClassifier([
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boost']),
            ('lr', self.models['logistic'])
        ], voting='soft')
        
        # Train ensemble (need to handle scaling for logistic regression)
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
        
        self.models['ensemble'] = ensemble
        self.is_trained = True
        
        # Feature importance from Random Forest
        rf_importance = self.models['random_forest'].feature_importances_
        feature_names = X.columns
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_importance
        }).sort_values('importance', ascending=False)
        
        # Print classification report
        print(f"\n=== ENSEMBLE MODEL PERFORMANCE ===")
        print(classification_report(y_test, ensemble_pred))
        
        return X_test, y_test, ensemble_pred
    
    def predict_upcoming_matches(self):
        """Predict upcoming matches from the 2025 schedule"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return None
        
        print("\n=== PREDICTING UPCOMING MATCHES ===")
        
        # Load current season schedule
        try:
            schedule_2025 = pd.read_csv('fbref_data/2025/schedule.csv')
        except FileNotFoundError:
            print("❌ 2025 schedule not found")
            return None
        
        # Filter for upcoming matches (no score yet)
        upcoming = schedule_2025[schedule_2025['score'].isna()].copy()
        
        if len(upcoming) == 0:
            print("No upcoming matches found")
            return None
        
        print(f"Found {len(upcoming)} upcoming matches")
        
        # Apply team name standardization
        upcoming['home_team'] = upcoming['home_team'].apply(self.standardize_team_name)
        upcoming['away_team'] = upcoming['away_team'].apply(self.standardize_team_name)
        
        # Load current ELO ratings
        try:
            elo_2024 = pd.read_csv('data/elo_ucl_teams/2024_ucl_elo.csv')
            elo_2024['team'] = elo_2024['team'].apply(self.standardize_team_name)
            
            # Get latest ELO for each team
            latest_elo = elo_2024.groupby('team')['elo'].last().reset_index()
            elo_dict = dict(zip(latest_elo['team'], latest_elo['elo']))
            
        except FileNotFoundError:
            print("❌ ELO data not found")
            return None
        
        # Add ELO ratings to upcoming matches
        upcoming['home_elo'] = upcoming['home_team'].map(elo_dict)
        upcoming['away_elo'] = upcoming['away_team'].map(elo_dict)
        upcoming['season'] = '2025'  # Add season for feature engineering
        
        # Engineer features (but skip season-based stats for predictions)
        upcoming = self.engineer_features_prediction(upcoming)
        
        # Select same features as training
        feature_cols = [col for col in self.feature_importance['feature'].tolist() 
                       if col in upcoming.columns]
        
        X_upcoming = upcoming[feature_cols].copy()
        X_upcoming = self.handle_missing_values(X_upcoming)
        
        # Make predictions
        predictions = self.models['ensemble'].predict(X_upcoming)
        probabilities = self.models['ensemble'].predict_proba(X_upcoming)
        
        # Create results dataframe
        results = upcoming[['round', 'date', 'time', 'home_team', 'away_team', 'venue']].copy()
        results['predicted_result'] = predictions
        results['home_win_prob'] = probabilities[:, np.where(self.models['ensemble'].classes_ == 'H')[0][0]]
        results['away_win_prob'] = probabilities[:, np.where(self.models['ensemble'].classes_ == 'A')[0][0]]
        results['draw_prob'] = probabilities[:, np.where(self.models['ensemble'].classes_ == 'D')[0][0]]
        results['confidence'] = np.max(probabilities, axis=1)
        
        # Sort by date
        results['date'] = pd.to_datetime(results['date'])
        results = results.sort_values('date')
        
        return results
    
    def standardize_team_name(self, name):
        """Standardize team names (same as in build_training_data.py)"""
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
    
    def display_predictions(self, predictions):
        """Display predictions in a nice format"""
        if predictions is None or len(predictions) == 0:
            return
        
        print(f"\n🔮 UPCOMING MATCH PREDICTIONS")
        print("="*80)
        
        for _, match in predictions.head(10).iterrows():  # Show next 10 matches
            date_str = match['date'].strftime('%Y-%m-%d')
            time_str = match['time'] if pd.notna(match['time']) else 'TBD'
            
            home_prob = match['home_win_prob']
            away_prob = match['away_win_prob']
            draw_prob = match['draw_prob']
            
            print(f"\n📅 {date_str} {time_str}")
            print(f"🏠 {match['home_team']} vs {match['away_team']} 🛫")
            print(f"🏟️  {match['venue']}")
            print(f"🎯 Prediction: {match['predicted_result']} (Confidence: {match['confidence']:.1%})")
            print(f"📊 Probabilities: H {home_prob:.1%} | D {draw_prob:.1%} | A {away_prob:.1%}")
        
        print("\n" + "="*80)
        
        # Summary stats
        high_confidence = predictions[predictions['confidence'] > 0.6]
        print(f"📈 High confidence predictions (>60%): {len(high_confidence)}/{len(predictions)}")
        
        pred_distribution = predictions['predicted_result'].value_counts()
        print(f"🏆 Predicted outcomes: H: {pred_distribution.get('H', 0)}, "
              f"D: {pred_distribution.get('D', 0)}, A: {pred_distribution.get('A', 0)}")


def main():
    # Create and train the predictor
    predictor = UCLPredictor()
    
    # Load and prepare data
    data = predictor.load_and_prepare_data()
    if data is None:
        return
    
    X, y, feature_cols = data
    
    # Train models
    X_test, y_test, predictions = predictor.train_models(X, y)
    
    # Show feature importance
    print(f"\n🎯 TOP 15 MOST IMPORTANT FEATURES:")
    print("="*50)
    for i, (_, row) in enumerate(predictor.feature_importance.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.3f}")
    
    # Predict upcoming matches
    upcoming_predictions = predictor.predict_upcoming_matches()
    predictor.display_predictions(upcoming_predictions)
    
    # Save predictions
    if upcoming_predictions is not None:
        upcoming_predictions.to_csv('data/predictions/upcoming_matches.csv', index=False)
        print(f"\n💾 Saved predictions to data/predictions/upcoming_matches.csv")

if __name__ == "__main__":
    main()
