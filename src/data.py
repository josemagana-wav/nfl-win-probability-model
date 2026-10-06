#%%
import nflreadpy as nfl

SEASONS = list(range(2016, 2026)) ##importing previous 10 seasons of data 

def load_data(seasons=SEASONS):
    schedules = nfl.load_schedules(seasons).to_pandas()
    pbp = nfl.load_pbp(seasons).to_pandas()

    #Take only regular season data, playoffs are far more unpredictable 
    schedules = schedules[schedules['game_type'] == 'REG']
    pbp = pbp[pbp['season_type'] == 'REG']
    return schedules, pbp


#%%
def clean_plays(pbp):
    plays = pbp[
        pbp['play_type'].isin(['run', 'pass']) &
        pbp['epa'].notna()
    ].copy()
    return plays


TEAM_FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
def build_target(schedules):
    # Drop games with missing scores (future/unplayed games)
    games = schedules.dropna(subset=['home_score', 'away_score']).copy()

    #Create binary target
    games['home_team'] = games['home_team'].replace(TEAM_FIX)
    games['away_team'] = games['away_team'].replace(TEAM_FIX)
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
    return games
