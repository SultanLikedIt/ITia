import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def explore_serie_a_data():
    """Explore the collected Serie A data"""
    
    # Load the data
    try:
        teams_df = pd.read_csv('serie_a_teams.csv')
        matches_df = pd.read_csv('serie_a_matches_2024.csv')
        print("✅ Data loaded successfully!")
    except FileNotFoundError:
        print("❌ Data files not found. Run data_collector.py first!")
        return
    
    print("\n" + "="*50)
    print("SERIE A TEAMS ANALYSIS")
    print("="*50)
    
    print(f"Total teams: {len(teams_df)}")
    print("\nTeams in Serie A 2024-25:")
    for i, team in enumerate(teams_df['name'], 1):
        print(f"{i:2d}. {team}")
    
    print("\n" + "="*50)
    print("MATCHES ANALYSIS")
    print("="*50)
    
    print(f"Total matches in dataset: {len(matches_df)}")
    
    # Check match status
    status_counts = matches_df['status'].value_counts()
    print(f"\nMatch status breakdown:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Analyze finished matches
    finished_matches = matches_df[matches_df['status'] == 'FINISHED'].copy()
    
    if len(finished_matches) > 0:
        print(f"\n📊 FINISHED MATCHES ANALYSIS ({len(finished_matches)} matches)")
        print("-" * 40)
        
        # Home vs Away wins
        winner_counts = finished_matches['winner'].value_counts()
        print("\nMatch outcomes:")
        for outcome, count in winner_counts.items():
            percentage = (count / len(finished_matches)) * 100
            print(f"  {outcome}: {count} ({percentage:.1f}%)")
        
        # Goals analysis
        if 'home_score' in finished_matches.columns:
            finished_matches['total_goals'] = finished_matches['home_score'] + finished_matches['away_score']
            finished_matches['goal_difference'] = abs(finished_matches['home_score'] - finished_matches['away_score'])
            
            print(f"\n⚽ GOALS ANALYSIS")
            print(f"Average goals per match: {finished_matches['total_goals'].mean():.2f}")
            print(f"Average goal difference: {finished_matches['goal_difference'].mean():.2f}")
            print(f"Highest scoring match: {finished_matches['total_goals'].max()} goals")
            
            # Show the highest scoring match
            highest_scoring = finished_matches[finished_matches['total_goals'] == finished_matches['total_goals'].max()]
            if len(highest_scoring) > 0:
                match = highest_scoring.iloc[0]
                print(f"  → {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}")
        
        # Team performance
        print(f"\n🏆 TEAM PERFORMANCE")
        print("-" * 20)
        
        # Count wins for each team (both home and away)
        team_wins = {}
        all_teams = set(finished_matches['home_team'].unique()) | set(finished_matches['away_team'].unique())
        
        for team in all_teams:
            home_wins = len(finished_matches[(finished_matches['home_team'] == team) & 
                                           (finished_matches['winner'] == 'HOME_TEAM')])
            away_wins = len(finished_matches[(finished_matches['away_team'] == team) & 
                                           (finished_matches['winner'] == 'AWAY_TEAM')])
            total_wins = home_wins + away_wins
            
            # Count total matches played
            home_matches = len(finished_matches[finished_matches['home_team'] == team])
            away_matches = len(finished_matches[finished_matches['away_team'] == team])
            total_matches = home_matches + away_matches
            
            if total_matches > 0:
                win_rate = (total_wins / total_matches) * 100
                team_wins[team] = {'wins': total_wins, 'matches': total_matches, 'win_rate': win_rate}
        
        # Sort teams by win rate
        sorted_teams = sorted(team_wins.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        print("Current form (by win rate):")
        for i, (team, stats) in enumerate(sorted_teams[:10], 1):
            print(f"{i:2d}. {team:<20} - {stats['wins']}/{stats['matches']} wins ({stats['win_rate']:.1f}%)")
        
        # Show some recent matches
        print(f"\n📅 RECENT MATCHES")
        print("-" * 20)
        finished_matches['date'] = pd.to_datetime(finished_matches['date'])
        recent_matches = finished_matches.nlargest(5, 'date')
        
        for _, match in recent_matches.iterrows():
            date_str = match['date'].strftime('%Y-%m-%d')
            print(f"{date_str}: {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}")
    
    else:
        print("No finished matches found yet. Season might just be starting!")
    
    # Show upcoming matches
    upcoming_matches = matches_df[matches_df['status'].isin(['SCHEDULED', 'TIMED'])].copy()
    if len(upcoming_matches) > 0:
        print(f"\n🔮 UPCOMING MATCHES ({len(upcoming_matches)} scheduled)")
        print("-" * 30)
        upcoming_matches['date'] = pd.to_datetime(upcoming_matches['date'])
        next_matches = upcoming_matches.nsmallest(5, 'date')
        
        for _, match in next_matches.iterrows():
            date_str = match['date'].strftime('%Y-%m-%d %H:%M')
            print(f"{date_str}: {match['home_team']} vs {match['away_team']}")
    
    print(f"\n✅ Data exploration complete!")
    print(f"📁 Your data files: serie_a_teams.csv, serie_a_matches_2024.csv")
    print(f"🎯 Ready for feature engineering and model building!")

if __name__ == "__main__":
    explore_serie_a_data()