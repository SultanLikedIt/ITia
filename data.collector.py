import requests
import pandas as pd
import json
from datetime import datetime
import time

class SerieADataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': api_key}
        
    def get_league_info(self):
        """Get Serie A basic info"""
        # Serie A competition code is 'SA' in football-data.org
        url = f"{self.base_url}/competitions/SA"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            competition = response.json()
            print(f"Found: {competition['name']} - Code: SA")
            return 'SA'
        else:
            print(f"Error accessing Serie A: {response.status_code}")
            return None
    
    def get_teams(self, competition_code):
        """Get all teams in the league"""
        url = f"{self.base_url}/competitions/{competition_code}/teams"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            teams_data = response.json()['teams']
            teams_df = pd.DataFrame([{
                'id': team['id'],
                'name': team['name'],
                'short_name': team['tla'],
                'venue': team['venue']
            } for team in teams_data])
            return teams_df
        else:
            print(f"Error: {response.status_code}")
            return None
    
    def get_matches(self, competition_code, season_year=2024):
        """Get all matches for a season"""
        url = f"{self.base_url}/competitions/{competition_code}/matches"
        params = {'season': season_year}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            matches_data = response.json()['matches']
            
            matches_list = []
            for match in matches_data:
                match_info = {
                    'id': match['id'],
                    'date': match['utcDate'],
                    'status': match['status'],
                    'matchday': match['matchday'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_team_id': match['homeTeam']['id'],
                    'away_team_id': match['awayTeam']['id']
                }
                
                # Add scores if match is finished
                if match['score']['fullTime']['home'] is not None:
                    match_info.update({
                        'home_score': match['score']['fullTime']['home'],
                        'away_score': match['score']['fullTime']['away'],
                        'winner': match['score']['winner']  # HOME_TEAM, AWAY_TEAM, or DRAW
                    })
                
                matches_list.append(match_info)
            
            return pd.DataFrame(matches_list)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def save_data(self, dataframe, filename):
        """Save data to CSV and pickle"""
        dataframe.to_csv(f'{filename}.csv', index=False)
        dataframe.to_pickle(f'{filename}.pkl')
        print(f"Data saved as {filename}.csv and {filename}.pkl")

# Usage example
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "19a17ec6f59d47bdae6c82568270482f"
    
    collector = SerieADataCollector(API_KEY)
    
    # Step 1: Find Serie A
    print("Accessing Serie A...")
    league_code = collector.get_league_info()
    
    if league_code:
        print(f"Using league code: {league_code}")
        
        # Step 2: Get teams
        print("Fetching Serie A teams...")
        teams_df = collector.get_teams(league_code)
        if teams_df is not None:
            print(f"Found {len(teams_df)} teams")
            print("Serie A teams:")
            print(teams_df[['name', 'short_name']].head(10))
            collector.save_data(teams_df, 'serie_a_teams')
        
        # Step 3: Get matches
        print("Fetching Serie A matches...")
        matches_df = collector.get_matches(league_code, 2024)
        if matches_df is not None:
            print(f"Found {len(matches_df)} matches")
            collector.save_data(matches_df, 'serie_a_matches_2024')
            
            # Show some basic stats
            finished_matches = matches_df[matches_df['status'] == 'FINISHED']
            print(f"\nFinished matches: {len(finished_matches)}")
            if len(finished_matches) > 0:
                print("Sample Serie A results:")
                print(finished_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head())
                
                # Show some interesting stats
                print(f"\nLeague standings preview:")
                home_wins = finished_matches['winner'].value_counts()
                print(home_wins)
    else:
        print("Serie A not accessible with this API key")