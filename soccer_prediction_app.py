import streamlit as st
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

st.title("Cyprus First Division AI Prediction ⚽")
st.write("Select teams for an AI-powered match prediction with odds!")

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

def api_request(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                st.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            elif response.status_code == 403:
                st.error("Invalid API key. Please check API_FOOTBALL_KEY in Streamlit Secrets.")
                return None
            else:
                raise e
    st.error(f"Failed after {retries} retries: {str(e)}")
    return None

def get_competition_id():
    try:
        url = f"{BASE_URL}leagues?country=Cyprus"
        response = api_request(url)
        if not response:
            st.error("No ID found for Cyprus First Division. Using fallback ID.")
            return 203
        leagues = response.json().get('response', [])
        for league in leagues:
            if league['league']['name'] == "1. Division":
                return league['league']['id']
        st.error("No ID found for Cyprus First Division. Using fallback ID.")
        return 203  # Fallback ID
    except Exception as e:
        st.error(f"Error fetching league ID: {str(e)}. Using fallback ID.")
        return 203

def fetch_odds(league_id, home_team_id, away_team_id, season):
    try:
        odds_url = f"{BASE_URL}odds?league={league_id}&season={season}&bookmaker=5&page=1"
        response = api_request(odds_url)
        if not response:
            return None
        odds_data = response.json().get('response', [])
        for match in odds_data:
            if match['fixture']['home']['id'] == home_team_id and match['fixture']['away']['id'] == away_team_id:
                for bet in match['bookmakers'][0]['bets']:
                    if bet['name'] == "Match Winner":
                        return {v['name']: v['odd'] for v in bet['values']}
        return None
    except Exception as e:
        st.error(f"Error fetching odds: {str(e)}")
        return None

def fetch_team_stats(home_team, away_team):
    try:
        league_id = get_competition_id()
        if not league_id:
            return None, None
        season = 2024
        teams_url = f"{BASE_URL}teams?league={league_id}&season={season}"
        response = api_request(teams_url)
        if not response:
            return None, None
        teams_data = response.json().get('response', [])
        team_ids = {team['team']['name']: team['team']['id'] for team in teams_data}
        
        home_api_name = team_mapping.get(home_team, home_team)
        away_api_name = team_mapping.get(away_team, away_team)
        if home_api_name not in team_ids or away_api_name not in team_ids:
            st.warning(f"Teams not found: {home_api_name} or {away_api_name}. Available: {', '.join(team_ids.keys())}")
            return None, None
        
        # Fetch team stats
        home_stats = api_request(f"{BASE_URL}teams/statistics?league={league_id}&season={season}&team={team_ids[home_api_name]}")
        away_stats = api_request(f"{BASE_URL}teams/statistics?league={league_id}&season={season}&team={team_ids[away_api_name]}")
        if not home_stats or not away_stats:
            return None, None
        home_stats = home_stats.json()['response']
        away_stats = away_stats.json()['response']
        
        # Fetch player stats
        players_url = f"{BASE_URL}players?league={league_id}&season={season}"
        home_players = api_request(f"{players_url}&team={team_ids[home_api_name]}")
        away_players = api_request(f"{players_url}&team={team_ids[away_api_name]}")
        if not home_players or not away_players:
            return None, None
        home_players = home_players.json()['response']
        away_players = away_players.json()['response']
        
        # Aggregate player stats
        home_goals = sum([p['statistics'][0]['goals']['total'] or 0 for p in home_players[:5]]) / 5
        away_goals = sum([p['statistics'][0]['goals']['total'] or 0 for p in away_players[:5]]) / 5
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
        
        # Fetch odds
        odds = fetch_odds(league_id, team_ids[home_api_name], team_ids[away_api_name], season)
        if odds:
            st.write(f"Betting Odds (Bookmaker 5): Home: {odds.get('Home', 'N/A')}, Draw: {odds.get('Draw', 'N/A')}, Away: {odds.get('Away', 'N/A')}")
        
        return {
            'home_goals': home_goals * 1.2,
            'away_goals': away_goals * 1.2,
            'home_form': home_form,
            'away_form': away_form,
            'home_possession': home_possession,
            'away_possession': away_possession,
            'home_shots': home_shots,
            'away_shots': away_shots
        }, odds
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None, None

if st.button("Predict Match"):
    stats, odds = fetch_team_stats(home_team, away_team)
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