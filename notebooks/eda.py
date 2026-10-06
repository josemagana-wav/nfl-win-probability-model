#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nflreadpy as nfl
from src.data import load_data, clean_plays, build_target
from src.features import build_features



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
#%%
sns.kdeplot(data=feature_df, x='off_epa_play_diff', hue='home_win',
            fill=True, common_norm=False, alpha=0.4)
plt.title('Offensive EPA/play differential, by game outcome')
# %%
sns.pairplot(feature_df[['off_epa_play_diff', 'def_epa_play_diff',
                         'home_win']], hue='home_win', diag_kind='kde')

#%%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))

from src.data import load_data, clean_plays, build_target
from src.features import build_features

#%%
schedules, pbp = load_data()
plays = clean_plays(pbp)
games = build_target(schedules)

#%%
games = build_features(plays, games)

#%%
print(len(games))
games[['home_team', 'week', 'home_form_off_epa']].head(20)
games[['off_epa_diff', 'def_epa_diff', 'home_field', 'rest_diff', 'div_game']].describe()


# %%
g0 = build_target(schedules)
print(len(g0))
print(g0.groupby('season').size())
# %%
out = build_features(plays, g0)
lost = g0[~g0['game_id'].isin(out['game_id'])]
print(len(lost))
print(lost[['season', 'week', 'home_team', 'away_team']].head(20))
print(lost.groupby('season').size())
