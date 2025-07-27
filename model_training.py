import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class SerieAPredictor:
    
    def __init__(self, features_file='serie_a_features.csv'):
        """Initialize the predictor with feature data"""
        print("Loading Serie A features...")
        self.df = pd.read_csv(features_file)
        self.label_encoder = LabelEncoder()
        
        print(f"Loaded {len(self.df)} matches with features")
        print(f"Features available: {len(self.df.columns)} columns")
        
    def prepare_data(self):
        """Prepare data for machine learning"""
        print("\nPreparing data for ML...")
        
        # Select feature columns (exclude match info and targets)
        feature_columns = [
            'home_form_wins', 'home_form_points', 'home_form_goals_for', 
            'home_form_goals_against', 'home_form_matches',
            'away_form_wins', 'away_form_points', 'away_form_goals_for',
            'away_form_goals_against', 'away_form_matches',
            'home_win_rate_at_home', 'home_avg_goals_at_home', 'home_matches_at_home',
            'away_win_rate_away', 'away_avg_goals_away', 'away_matches_away',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_avg_goals', 'h2h_matches'
        ]
        
        # Create feature matrix X
        self.X = self.df[feature_columns].copy()
        
        # Create target variable y (what we want to predict)
        self.y = self.df['winner'].copy()
        
        # Encode target labels: HOME_TEAM=0, DRAW=1, AWAY_TEAM=2
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target distribution:")
        target_counts = pd.Series(self.y).value_counts()
        for outcome, count in target_counts.items():
            percentage = (count / len(self.y)) * 100
            print(f"  {outcome}: {count} ({percentage:.1f}%)")
        
        return self.X, self.y_encoded
    
    def train_models(self):
        """Train multiple ML models and compare them"""
        print("\n🤖 Training machine learning models...")
        
        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        print(f"Training set: {len(X_train)} matches")
        print(f"Test set: {len(X_test)} matches")
        
        # Define models to try
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.trained_models = {}
        self.model_scores = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            self.trained_models[name] = model
            self.model_scores[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'test_labels': y_test
            }
            
            print(f"{name} Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Find best model
        best_model_name = max(self.model_scores.keys(), key=lambda x: self.model_scores[x]['accuracy'])
        self.best_model = self.trained_models[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name} with {self.model_scores[best_model_name]['accuracy']:.3f} accuracy")
        
        return X_test, y_test
    
    def analyze_predictions(self, X_test, y_test):
        """Analyze how well our best model performs"""
        best_model_name = max(self.model_scores.keys(), key=lambda x: self.model_scores[x]['accuracy'])
        y_pred = self.model_scores[best_model_name]['predictions']
        
        print(f"\n📊 Detailed Analysis for {best_model_name}:")
        print("="*50)
        
        # Classification report
        target_names = ['HOME_TEAM', 'DRAW', 'AWAY_TEAM']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓")
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm_df)
        
        # Feature importance (if Random Forest or Gradient Boosting)
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\n🎯 Most Important Features for Predictions:")
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(feature_importance.head(10))
    
    def make_sample_predictions(self):
        """Make predictions on some recent matches to see how it works"""
        print(f"\n🔮 Sample Predictions:")
        print("="*50)
        
        # Get some recent matches from our test set
        recent_matches = self.df.tail(10).copy()
        
        for idx, match in recent_matches.iterrows():
            # Prepare features for this match
            features = match[self.X.columns].values.reshape(1, -1)
            
            # Make prediction
            prediction = self.best_model.predict(features)[0]
            probabilities = self.best_model.predict_proba(features)[0]
            
            # Convert back to readable format
            predicted_outcome = self.label_encoder.inverse_transform([prediction])[0]
            actual_outcome = match['winner']
            
            # Show results
            home_prob = probabilities[0] * 100
            draw_prob = probabilities[1] * 100  
            away_prob = probabilities[2] * 100
            
            print(f"\n{match['home_team']} vs {match['away_team']}")
            print(f"Actual: {actual_outcome}")
            print(f"Predicted: {predicted_outcome}")
            print(f"Probabilities: Home {home_prob:.1f}% | Draw {draw_prob:.1f}% | Away {away_prob:.1f}%")
            
            # Check if prediction was correct
            correct = "✅" if predicted_outcome == actual_outcome else "❌"
            print(f"Result: {correct}")
    
    def save_model(self, filename='serie_a_predictor.pkl'):
        """Save the trained model for later use"""
        model_data = {
            'model': self.best_model,
            'label_encoder': self.label_encoder,
            'feature_columns': list(self.X.columns),
            'accuracy': max(self.model_scores.values(), key=lambda x: x['accuracy'])['accuracy']
        }
        
        joblib.dump(model_data, filename)
        print(f"\n💾 Model saved as {filename}")
        print(f"Model accuracy: {model_data['accuracy']:.3f}")

# Main execution
if __name__ == "__main__":
    # Create predictor
    predictor = SerieAPredictor()
    
    # Prepare data
    X, y = predictor.prepare_data()
    
    # Train models
    X_test, y_test = predictor.train_models()
    
    # Analyze results
    predictor.analyze_predictions(X_test, y_test)
    
    # Show sample predictions
    predictor.make_sample_predictions()
    
    # Save the best model
    predictor.save_model()
    
    print(f"\n🎯 Model training complete!")
    print(f"📈 Ready to predict Serie A matches!")
    print(f"💾 Model saved for web app integration")