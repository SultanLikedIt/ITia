import pandas as pd
import numpy as np
from datetime import datetime

class SerieAFeatureEngineer:
    
    def __init__(self, data_file='serie_a_all_seasons.csv'):
        """Load and prepare Serie A data for feature engineering"""
        print("Loading Serie A data...")
        self.df = pd.read_csv(data_file)
        
        # Clean and prepare data
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Only use finished matches for training
        self.finished_matches = self.df[self.df['status'] == 'FINISHED'].copy()
        print(f"Loaded {len(self.finished_matches)} finished matches for feature engineering")
        
    def calculate_team_form(self, team, date, num_matches=5):
        """Calculate recent form for a team before a specific date"""
        
        # Get recent matches for this team before the given date
        team_matches = self.finished_matches[
            ((self.finished_matches['home_team'] == team) | 
             (self.finished_matches['away_team'] == team)) &
            (self.finished_matches['date'] < date)
        ].tail(num_matches)
        
        if len(team_matches) == 0:
            return {'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0, 'points': 0, 'matches_played': 0}
        
        wins = draws = losses = goals_for = goals_against = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                # Team played at home
                goals_for += match['home_score']
                goals_against += match['away_score']
                
                if match['winner'] == 'HOME_TEAM':
                    wins += 1
                elif match['winner'] == 'DRAW':
                    draws += 1
                else:
                    losses += 1
            else:
                # Team played away
                goals_for += match['away_score']
                goals_against += match['home_score']
                
                if match['winner'] == 'AWAY_TEAM':
                    wins += 1
                elif match['winner'] == 'DRAW':
                    draws += 1
                else:
                    losses += 1
        
        points = wins * 3 + draws * 1
        
        return {
            'wins': wins,
            'draws': draws, 
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'points': points,
            'matches_played': len(team_matches)
        }
    
    def calculate_head_to_head(self, home_team, away_team, date, num_matches=5):
        """Calculate head-to-head record between two teams"""
        
        h2h_matches = self.finished_matches[
            (((self.finished_matches['home_team'] == home_team) & 
              (self.finished_matches['away_team'] == away_team)) |
             ((self.finished_matches['home_team'] == away_team) & 
              (self.finished_matches['away_team'] == home_team))) &
            (self.finished_matches['date'] < date)
        ].tail(num_matches)
        
        if len(h2h_matches) == 0:
            return {'home_wins': 0, 'draws': 0, 'away_wins': 0, 'avg_goals': 0, 'h2h_matches': 0}
        
        home_wins = draws = away_wins = total_goals = 0
        
        for _, match in h2h_matches.iterrows():
            total_goals += match['home_score'] + match['away_score']
            
            if match['home_team'] == home_team:
                # Current home team was home in this H2H match
                if match['winner'] == 'HOME_TEAM':
                    home_wins += 1
                elif match['winner'] == 'DRAW':
                    draws += 1
                else:
                    away_wins += 1
            else:
                # Current home team was away in this H2H match
                if match['winner'] == 'AWAY_TEAM':
                    home_wins += 1
                elif match['winner'] == 'DRAW':
                    draws += 1
                else:
                    away_wins += 1
        
        return {
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
            'avg_goals': total_goals / len(h2h_matches) if len(h2h_matches) > 0 else 0,
            'h2h_matches': len(h2h_matches)
        }
    
    def calculate_home_away_advantage(self, team, date, venue='home'):
        """Calculate team's performance at home or away"""
        
        if venue == 'home':
            team_matches = self.finished_matches[
                (self.finished_matches['home_team'] == team) &
                (self.finished_matches['date'] < date)
            ]
        else:
            team_matches = self.finished_matches[
                (self.finished_matches['away_team'] == team) &
                (self.finished_matches['date'] < date)
            ]
        
        if len(team_matches) == 0:
            return {'win_rate': 0, 'avg_goals_for': 0, 'avg_goals_against': 0, 'matches_played': 0}
        
        wins = 0
        total_goals_for = total_goals_against = 0
        
        for _, match in team_matches.iterrows():
            if venue == 'home':
                total_goals_for += match['home_score']
                total_goals_against += match['away_score']
                if match['winner'] == 'HOME_TEAM':
                    wins += 1
            else:
                total_goals_for += match['away_score']
                total_goals_against += match['home_score']
                if match['winner'] == 'AWAY_TEAM':
                    wins += 1
        
        return {
            'win_rate': wins / len(team_matches),
            'avg_goals_for': total_goals_for / len(team_matches),
            'avg_goals_against': total_goals_against / len(team_matches),
            'matches_played': len(team_matches)
        }
    
    def create_features(self):
        """Create all features for machine learning"""
        
        print("Creating features for machine learning...")
        features_list = []
        
        for idx, match in self.finished_matches.iterrows():
            if idx % 100 == 0:
                print(f"Processing match {idx}/{len(self.finished_matches)}")
            
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = match['date']
            
            # Calculate features
            home_form = self.calculate_team_form(home_team, match_date)
            away_form = self.calculate_team_form(away_team, match_date)
            
            home_advantage = self.calculate_home_away_advantage(home_team, match_date, 'home')
            away_advantage = self.calculate_home_away_advantage(away_team, match_date, 'away')
            
            h2h = self.calculate_head_to_head(home_team, away_team, match_date)
            
            # Create feature row
            features = {
                # Match info
                'match_id': match['id'],
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'season': match['season'],
                
                # Home team form
                'home_form_wins': home_form['wins'],
                'home_form_points': home_form['points'],
                'home_form_goals_for': home_form['goals_for'],
                'home_form_goals_against': home_form['goals_against'],
                'home_form_matches': home_form['matches_played'],
                
                # Away team form  
                'away_form_wins': away_form['wins'],
                'away_form_points': away_form['points'],
                'away_form_goals_for': away_form['goals_for'],
                'away_form_goals_against': away_form['goals_against'],
                'away_form_matches': away_form['matches_played'],
                
                # Home advantage
                'home_win_rate_at_home': home_advantage['win_rate'],
                'home_avg_goals_at_home': home_advantage['avg_goals_for'],
                'home_matches_at_home': home_advantage['matches_played'],
                
                # Away advantage
                'away_win_rate_away': away_advantage['win_rate'],
                'away_avg_goals_away': away_advantage['avg_goals_for'],
                'away_matches_away': away_advantage['matches_played'],
                
                # Head to head
                'h2h_home_wins': h2h['home_wins'],
                'h2h_draws': h2h['draws'],
                'h2h_away_wins': h2h['away_wins'],
                'h2h_avg_goals': h2h['avg_goals'],
                'h2h_matches': h2h['h2h_matches'],
                
                # Target variables (what we want to predict)
                'home_score': match['home_score'],
                'away_score': match['away_score'],
                'winner': match['winner']  # HOME_TEAM, DRAW, AWAY_TEAM
            }
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Remove matches with insufficient data (early season matches)
        features_df = features_df[
            (features_df['home_form_matches'] >= 3) & 
            (features_df['away_form_matches'] >= 3)
        ].reset_index(drop=True)
        
        print(f"Created features for {len(features_df)} matches")
        return features_df
    
    def save_features(self, features_df, filename='serie_a_features.csv'):
        """Save features to file"""
        features_df.to_csv(filename, index=False)
        features_df.to_pickle(filename.replace('.csv', '.pkl'))
        print(f"Features saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Create feature engineer
    engineer = SerieAFeatureEngineer()
    
    # Create features for all matches
    features_df = engineer.create_features()
    
    # Show sample features
    print("\n📊 SAMPLE FEATURES:")
    print(features_df[['home_team', 'away_team', 'home_form_points', 'away_form_points', 'winner']].head())
    
    # Save features
    engineer.save_features(features_df)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"📁 Features saved as: serie_a_features.csv")
    print(f"🎯 Ready for model training with {len(features_df)} matches!")