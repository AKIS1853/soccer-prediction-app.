import streamlit as st
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

st.title("Cyprus First Division AI Prediction ⚽")
st.write("Select teams for an AI-powered match prediction!")

API_KEY = st.secrets["API_FOOTBALL_KEY"]
HEADERS = {"X-RapidAPI-Key": API_KEY, "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"}
BASE_URL = "https://api-football-v1.p.rapidapi.com/v3/"

teams = ["APOEL Nicosia", "Aris Limassol", "Omonia Nicosia", "Paphos FC", "AEK Larnaca", "Anorthosis Famagusta"]
team_mapping = {
    "APOEL Nicosia": "APOEL Nicosia FC",
    "Aris Limassol": "Aris Limassol FC",
    "Omonia Nicosia": "Omonia Nicosia",
    "Paphos FC": "Paphos FC",
    "AEK Larnaca": "AEK Larnaca",
    "Anorthosis Famagusta": "Anorthosis Famagusta"
}

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)

# Fallback training data
data = {
    'home_goals': [2.5, 1.2, 3.0, 0.8, 2.0, 1.5, 2.8, 1.0],
    'away_goals': [1.0, 2.0, 0.5, 1.5, 1.2, 1.8, 0.9, 1.3],
    'home_form': [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.85, 0.3],
    'away_form': [0.5, 0.7, 0.3, 0.6, 0.5, 0.65, 0.4, 0.55],
    'home_possession': [55, 50, 52, 48, 53, 51, 54, 49],
    'away_possession': [50, 52, 48, 51, 49, 50, 47, 52],
    'home_shots': [4.5, 3.8, 5.0, 3.0, 4.2, 3.5, 4.8, 3.2],
    'away_shots': [3.5, 4.0, 3.0, 3.8, 3.2, 3.6, 3.1, 3.9],
    'result': ['home_win', 'away_win', 'home_win', 'draw', 'home_win', 'draw', 'home_win', 'away_win']
}
df = pd.DataFrame(data)
X = df[['home_goals', 'away_goals', 'home_form', 'away_form', 'home_possession', 'away_possession', 'home_shots', 'away_shots']]
y = df['result']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(multi_class='multinomial', random_state=42)
model.fit(X_scaled, y)

def get_competition_id():
    try:
        url = f"{BASE_URL}leagues?country=Cyprus"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        leagues = response.json().get('response', [])
        for league in leagues:
            if league['league']['name'] == "1. Division":
                return league['league']['id']
        st.error("No ID found for Cyprus First Division. Using fallback ID.")
        return 203  # Fallback ID for Cyprus First Division
    except Exception as e:
        st.error(f"Error fetching league ID: {str(e)}. Using fallback ID.")
        return 203

def fetch_team_stats(home_team, away_team):
    try:
        league_id = get_competition_id()
        if not league_id:
            return None
        season = 2024  # Adjust for 2024–25 season
        url = f"{BASE_URL}teams/statistics?league={league_id}&season={season}"
        home_api_name = team_mapping.get(home_team, home_team)
        away_api_name = team_mapping.get(away_team, away_team)
        
        # Fetch team IDs
        teams_url = f"{BASE_URL}teams?league={league_id}&season={season}"
        teams_response = requests.get(teams_url, headers=HEADERS)
        teams_response.raise_for_status()
        teams_data = teams_response.json().get('response', [])
        team_ids = {}
        for team in teams_data:
            team_name = team['team']['name']
            team_ids[team_name] = team['team']['id']
        
        if home_api_name not in team_ids or away_api_name not in team_ids:
            st.warning(f"Teams not found: {home_api_name} or {away_api_name}.")
            return None
        
        # Fetch team stats
        home_stats = requests.get(f"{BASE_URL}teams/statistics?league={league_id}&season={season}&team={team_ids[home_api_name]}", headers=HEADERS).json()['response']
        away_stats = requests.get(f"{BASE_URL}teams/statistics?league={league_id}&season={season}&team={team_ids[away_api_name]}", headers=HEADERS).json()['response']
        
        # Fetch player stats
        players_url = f"{BASE_URL}players?league={league_id}&season={season}"
        home_players = requests.get(f"{players_url}&team={team_ids[home_api_name]}", headers=HEADERS).json()['response']
        away_players = requests.get(f"{players_url}&team={team_ids[away_api_name]}", headers=HEADERS).json()['response']
        
        # Aggregate player stats (e.g., top scorer goals)
        home_goals = sum([p['statistics'][0]['goals']['total'] or 0 for p in home_players[:5]]) / 5  # Top 5 players
        away_goals = sum([p['statistics'][0]['goals']['total'] or 0 for p in away_players[:5]]) / 5
        
        # Team stats
        home_form = float(home_stats['form'].count('W')) / max(len(home_stats['form']), 1)
        away_form = float(away_stats['form'].count('W')) / max(len(away_stats['form']), 1)
        home_possession = float(home_stats['fixtures']['possession']['average'] or 50)
        away_possession = float(away_stats['fixtures']['possession']['average'] or 50)
        home_shots = float(home_stats['fixtures']['shots']['on'] or 4)
        away_shots = float(away_stats['fixtures']['shots']['on'] or 4)
        
        # Key players
        home_top_player = max(home_players, key=lambda p: p['statistics'][0]['goals']['total'] or 0, default={'player': {'name': 'Unknown'}})
        away_top_player = max(away_players, key=lambda p: p['statistics'][0]['goals']['total'] or 0, default={'player': {'name': 'Unknown'}})
        
        st.write(f"Key Players - {home_team}: {home_top_player['player']['name']} ({home_top_player['statistics'][0]['goals']['total']} goals)")
        st.write(f"Key Players - {away_team}: {away_top_player['player']['name']} ({away_top_player['statistics'][0]['goals']['total']} goals)")
        
        return {
            'home_goals': home_goals * 1.2,  # Boost player goals
            'away_goals': away_goals * 1.2,
            'home_form': home_form,
            'away_form': away_form,
            'home_possession': home_possession,
            'away_possession': away_possession,
            'home_shots': home_shots,
            'away_shots': away_shots
        }
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

if st.button("Predict Match"):
    stats = fetch_team_stats(home_team, away_team)
    if stats is None:
        st.warning("No data found. Using fallback stats.")
        stats = {'home_goals': 1.0, 'away_goals': 1.0, 'home_form': 0.5, 'away_form': 0.5, 
                 'home_possession': 50, 'away_possession': 50, 'home_shots': 4, 'away_shots': 4}
    
    new_match = [[stats['home_goals'], stats['away_goals'], stats['home_form'], stats['away_form'], 
                  stats['home_possession'], stats['away_possession'], stats['home_shots'], stats['away_shots']]]
    new_match_scaled = scaler.transform(new_match)
    prediction = model.predict(new_match_scaled)[0]
    prob = model.predict_proba(new_match_scaled)[0]
    prob_dict = {model.classes_[i]: round(prob[i] * 100, 2) for i in range(len(prob))}
    
    st.success(f"Prediction: **{prediction.replace('home_win', 'Home Win').replace('away_win', 'Away Win').replace('draw', 'Draw')}**")
    st.write(f"Home Win ({home_team}): {prob_dict['home_win']}%")
    st.write(f"Draw: {prob_dict['draw']}%")
    st.write(f"Away Win ({away_team}): {prob_dict['away_win']}%")