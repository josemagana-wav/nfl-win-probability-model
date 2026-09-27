#%%
import pandas as pd

#%%
df = pd.DataFrame({
    'team':  ['NE', 'NE', 'NE', 'KC', 'KC', 'KC'],
    'week':  [1, 2, 3, 1, 2, 3],
    'epa':   [0.10, -0.05, 0.20, 0.30, 0.25, 0.40],
})

#%%
df.groupby('team')['epa'].mean()


# %%

schedule = pd.DataFrame({'game_id': [1, 2], 'home_team': ['NE' , 'KC']})
team_stats = pd.DataFrame({'home_team': ['NE', 'KC'], 'off_epa': [0.08, 0.32]})

schedule.merge(team_stats, on='home_team', how='left')
# %%
df['avg_team_epa'] = df.groupby('team')['epa'].transform('mean')
df
# %%
s = pd.Series([10,20,30,40], index=['wk1','wk2','wk3','wk4'])
s.shift(1).rolling(2, min_periods=1).mean()
# %%
s = pd.Series([10,20,30,40,50])
s.shift(1).rolling(2, min_periods=1).mean()
# %%
