from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict
import uvicorn

# Load the trained model
try:
    model_data = joblib.load('serie_a_predictor.pkl')
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_columns = model_data['feature_columns']
    model_accuracy = model_data['accuracy']
    print(f"✅ Model loaded successfully! Accuracy: {model_accuracy:.3f}")
except FileNotFoundError:
    print("❌ Model file not found. Run model_training.py first!")
    model = None

# Load team data and features for predictions
try:
    teams_df = pd.read_csv('serie_a_teams.csv')
    features_df = pd.read_csv('serie_a_features.csv')
    print(f"✅ Data loaded: {len(teams_df)} teams, {len(features_df)} historical matches")
except FileNotFoundError:
    print("❌ Data files not found!")
    teams_df = None
    features_df = None

# Create FastAPI app
app = FastAPI(
    title="Serie A Match Predictor",
    description="AI-powered Serie A football match outcome prediction",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str

class TeamForm(BaseModel):
    wins: int
    points: int
    goals_for: int
    goals_against: int
    matches: int

class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    home_form: TeamForm
    away_form: TeamForm

def get_team_recent_form(team_name: str, num_matches: int = 5) -> Dict:
    """Calculate recent form for a team using latest available data"""
    if features_df is None:
        return {'wins': 0, 'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0}
    
    # Get recent matches for this team (both home and away)
    team_matches = features_df[
        (features_df['home_team'] == team_name) | 
        (features_df['away_team'] == team_name)
    ].tail(num_matches)
    
    if len(team_matches) == 0:
        return {'wins': 0, 'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0}
    
    # Use the most recent match's form data
    recent_match = team_matches.iloc[-1]
    
    if recent_match['home_team'] == team_name:
        return {
            'wins': recent_match['home_form_wins'],
            'points': recent_match['home_form_points'],
            'goals_for': recent_match['home_form_goals_for'],
            'goals_against': recent_match['home_form_goals_against'],
            'matches': recent_match['home_form_matches']
        }
    else:
        return {
            'wins': recent_match['away_form_wins'],
            'points': recent_match['away_form_points'],
            'goals_for': recent_match['away_form_goals_for'],
            'goals_against': recent_match['away_form_goals_against'],
            'matches': recent_match['away_form_matches']
        }

def calculate_features_for_prediction(home_team: str, away_team: str) -> np.ndarray:
    """Calculate features for a new match prediction"""
    
    # Get recent form for both teams
    home_form = get_team_recent_form(home_team)
    away_form = get_team_recent_form(away_team)
    
    # Get home/away advantage (simplified using recent data)
    home_matches = features_df[features_df['home_team'] == home_team]
    away_matches = features_df[features_df['away_team'] == away_team]
    
    if len(home_matches) > 0:
        home_win_rate = home_matches['home_win_rate_at_home'].iloc[-1]
        home_avg_goals = home_matches['home_avg_goals_at_home'].iloc[-1]
        home_matches_played = home_matches['home_matches_at_home'].iloc[-1]
    else:
        home_win_rate = home_avg_goals = home_matches_played = 0
    
    if len(away_matches) > 0:
        away_win_rate = away_matches['away_win_rate_away'].iloc[-1]
        away_avg_goals = away_matches['away_avg_goals_away'].iloc[-1]
        away_matches_played = away_matches['away_matches_away'].iloc[-1]
    else:
        away_win_rate = away_avg_goals = away_matches_played = 0
    
    # Get head-to-head (simplified)
    h2h_matches = features_df[
        ((features_df['home_team'] == home_team) & (features_df['away_team'] == away_team)) |
        ((features_df['home_team'] == away_team) & (features_df['away_team'] == home_team))
    ]
    
    if len(h2h_matches) > 0:
        recent_h2h = h2h_matches.iloc[-1]
        h2h_home_wins = recent_h2h['h2h_home_wins']
        h2h_draws = recent_h2h['h2h_draws']
        h2h_away_wins = recent_h2h['h2h_away_wins']
        h2h_avg_goals = recent_h2h['h2h_avg_goals']
        h2h_matches_count = recent_h2h['h2h_matches']
    else:
        h2h_home_wins = h2h_draws = h2h_away_wins = h2h_avg_goals = h2h_matches_count = 0
    
    # Create feature vector matching training data
    features = np.array([
        home_form['wins'], home_form['points'], home_form['goals_for'],
        home_form['goals_against'], home_form['matches'],
        away_form['wins'], away_form['points'], away_form['goals_for'],
        away_form['goals_against'], away_form['matches'],
        home_win_rate, home_avg_goals, home_matches_played,
        away_win_rate, away_avg_goals, away_matches_played,
        h2h_home_wins, h2h_draws, h2h_away_wins, h2h_avg_goals, h2h_matches_count
    ]).reshape(1, -1)
    
    return features, home_form, away_form

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Serie A Match Predictor API",
        "model_accuracy": f"{model_accuracy:.1%}" if model else "Model not loaded",
        "status": "ready" if model else "model_missing"
    }

@app.get("/teams")
async def get_teams():
    """Get list of all Serie A teams"""
    if teams_df is None:
        raise HTTPException(status_code=500, detail="Teams data not available")
    
    teams = teams_df['name'].tolist()
    return {"teams": teams, "count": len(teams)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: MatchPredictionRequest):
    """Predict outcome of a Serie A match"""
    
    if model is None or features_df is None:
        raise HTTPException(status_code=500, detail="Model or data not available")
    
    try:
        # Calculate features for this match
        features, home_form, away_form = calculate_features_for_prediction(
            request.home_team, request.away_team
        )
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Convert prediction back to readable format
        predicted_outcome = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence (highest probability)
        confidence = max(probabilities)
        
        # Create probability dictionary
        prob_dict = {
            "HOME_TEAM": float(probabilities[0]),
            "DRAW": float(probabilities[1]),
            "AWAY_TEAM": float(probabilities[2])
        }
        
        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            prediction=predicted_outcome,
            confidence=float(confidence),
            probabilities=prob_dict,
            home_form=TeamForm(**home_form),
            away_form=TeamForm(**away_form)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/stats")
async def get_model_stats():
    """Get model statistics and info"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    return {
        "model_type": type(model).__name__,
        "accuracy": f"{model_accuracy:.1%}",
        "features_used": len(feature_columns),
        "training_matches": len(features_df) if features_df is not None else 0,
        "available_teams": len(teams_df) if teams_df is not None else 0
    }

# Run the server
if __name__ == "__main__":
    if model is None:
        print("⚠️  Model not found! Run model_training.py first")
    else:
        print("🚀 Starting Serie A Predictor API...")
        print("📊 Model accuracy:", f"{model_accuracy:.1%}")
        print("🔗 API will be available at: http://localhost:8000")
        print("📖 API docs at: http://localhost:8000/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)