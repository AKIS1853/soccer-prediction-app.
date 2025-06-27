import streamlit as st
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
st.title("Πρόβλεψη Αγώνα Ποδοσφαίρου AI ⚽")
st.write("Επιλέξτε ομάδες και δείτε την πρόβλεψη!")
API_KEY = st.secrets["FOOTBALL_API_KEY"]
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "http://api.football-data.org/v4/"
teams = ["ΠΑΟΚ", "ΑΕΚ", "Ολυμπιακός", "Παναθηναϊκός"]
home_team = st.selectbox("Ομάδα Γηπεδούχου", teams)
away_team = st.selectbox("Ομάδα Φιλοξενούμενου", teams)
data = {
    'home_goals': [2.5, 1.2, 3.0, 0.8, 2.0, 1.5, 2.8, 1.0],
    'away_goals': [1.0, 2.0, 0.5, 1.5, 1.2, 1.8, 0.9, 1.3],
    'home_form': [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.85, 0.3],
    'away_form': [0.5, 0.7, 0.3, 0.6, 0.5, 0.65, 0.4, 0.55],
    'result': ['home_win', 'away_win', 'home_win', 'draw', 'home_win', 'draw', 'home_win', 'away_win']
}
df = pd.DataFrame(data)
X = df[['home_goals', 'away_goals', 'home_form', 'away_form']]
y = df['result']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(multi_class='multinomial', random_state=42)
model.fit(X_scaled, y)
def fetch_team_stats(home_team, away_team):
    try:
        url = f"{BASE_URL}competitions/2016/matches?status=FINISHED&limit=10"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        matches = response.json().get('matches', [])
        home_goals, away_goals, home_form, away_form = [], [], [], []
        for match in matches:
            home = match['homeTeam']['name']
            away = match['awayTeam']['name']
            score = match['score']['fullTime']
            if home in teams and away in teams:
                home_goals.append(score['home'] or 0)
                away_goals.append(score['away'] or 0)
                home_form.append(0.8 if score['home'] > score['away'] else 0.5 if score['home'] == score['away'] else 0.3)
                away_form.append(0.8 if score['away'] > score['home'] else 0.5 if score['away'] == score['home'] else 0.3)
        if home_goals and away_goals:
            return {
                'home_goals': np.mean(home_goals),
                'away_goals': np.mean(away_goals),
                'home_form': np.mean(home_form),
                'away_form': np.mean(away_form)
            }
        else:
            return None
    except:
        return None
if st.button("Πρόβλεψη"):
    stats = fetch_team_stats(home_team, away_team)
    if stats is None:
        st.warning("Δεν βρέθηκαν δεδομένα. Χρησιμοποιώ προεπιλεγμένα δεδομένα.")
        stats = {'home_goals': 1.0, 'away_goals': 1.0, 'home_form': 0.5, 'away_form': 0.5}
    new_match = [[stats['home_goals'], stats['away_goals'], stats['home_form'], stats['away_form']]]
    new_match_scaled = scaler.transform(new_match)
    prediction = model.predict(new_match_scaled)[0]
    prob = model.predict_proba(new_match_scaled)[0]
    prob_dict = {model.classes_[i]: round(prob[i] * 100, 2) for i in range(len(prob))}
    st.success(f"Αποτέλεσμα: **{prediction.replace('home_win', 'Νίκη Γηπεδούχου').replace('away_win', 'Νίκη Φιλοξενούμενου').replace('draw', 'Ισοπαλία')}**")
    st.write(f"Νίκη Γηπεδούχου: {prob_dict['home_win']}%")
    st.write(f"Ισοπαλία: {prob_dict['draw']}%")
    st.write(f"Νίκη Φιλοξενούμενου: {prob_dict['away_win']}%")