#%%
import nflreadpy as nfl

#%%
SEASONS = list(range(2015, 2025)) ##importing previous 10 seasons of data 

#%%
schedules = nfl.load_schedules(SEASONS).to_pandas()
pbp = nfl.load_pbp(SEASONS).to_pandas()

#Take only regular season data, playoffs are far more unpredictable 
schedules = schedules[schedules['game_type'] == 'REG']
pbp = pbp[pbp['season_type'] == 'REG']


print(pbp['epa'].describe())
print(pbp[pbp['play_type'] == 'run']['epa'].mean())
# %%
