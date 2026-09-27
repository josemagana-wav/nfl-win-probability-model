#%%
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

# Drop games with missing scores (future/unplayed games)
games = schedules.dropna(subset=['home_score', 'away_score']).copy()

#Create binary target
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

