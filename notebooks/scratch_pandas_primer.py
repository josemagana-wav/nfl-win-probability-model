#%%
import pandas as pd

#%%
df = pd.DataFrame({
    'team':  ['NE', 'NE', 'NE', 'KC', 'KC', 'KC'],
    'week':  [1, 2, 3, 1, 2, 3],
    'epa':   [0.10, -0.05, 0.20, 0.30, 0.25, 0.40],
})

df.groupby('team')['epa'].mean()


# %%
