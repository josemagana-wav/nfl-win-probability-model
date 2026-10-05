#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nflreadpy as nfl



SEASONS = list(range(2015, 2025)) ##importing previous 10 seasons of data 


schedules = nfl.load_schedules(SEASONS).to_pandas()
pbp = nfl.load_pbp(SEASONS).to_pandas()

#Take only regular season data, playoffs are far more unpredictable 
schedules = schedules[schedules['game_type'] == 'REG']
pbp = pbp[pbp['season_type'] == 'REG']


#%%
#Remove plays such as special teams, spikes and kneels
plays = pbp[
    pbp['play_type'].isin(['run', 'pass']) &
    pbp['epa'].notna()
].copy()

#%%
off_epa = (
    plays.groupby(['game_id', 'posteam'])['epa']
    .mean()
    .reset_index()
    .rename(columns={'posteam': 'team', 'epa': 'off_epa_play'})
)

#%%
def_epa = (
    plays.groupby(['game_id', 'defteam'])['epa']
    .mean()
    .reset_index()
    .rename(columns={'defteam': 'team', 'epa': 'def_epa_play'})
)
#%%
games = schedules.dropna(subset=['home_score', 'away_score']).copy()

games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
#%%
feature_df = games[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_win']].copy()

feature_df = feature_df.merge(
    off_epa.rename(columns={'team': 'home_team', 'off_epa_play': 'home_off_epa_play'}),
    on=['game_id', 'home_team'])
feature_df = feature_df.merge(
    off_epa.rename(columns={'team': 'away_team', 'off_epa_play': 'away_off_epa_play'}),
    on=['game_id', 'away_team'])
feature_df = feature_df.merge(
    def_epa.rename(columns={'team': 'home_team', 'def_epa_play': 'home_def_epa_play'}),
    on=['game_id', 'home_team'])
feature_df = feature_df.merge(
    def_epa.rename(columns={'team': 'away_team', 'def_epa_play': 'away_def_epa_play'}),
    on=['game_id', 'away_team'])

feature_df['off_epa_play_diff'] = feature_df['home_off_epa_play'] - feature_df['away_off_epa_play']
feature_df['def_epa_play_diff'] = feature_df['home_def_epa_play'] - feature_df['away_def_epa_play']

#%%
corr = feature_df[['off_epa_play_diff', 'def_epa_play_diff', 'home_win']].corr(numeric_only = True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0)
plt.title('Correlation between candidate features and home wins')
plt.tight_layout()
plt.savefig('correlation heatmap.png')
# %%
