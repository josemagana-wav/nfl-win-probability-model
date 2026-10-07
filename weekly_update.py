import pandas as pd
import nflreadpy as nfl

LOG = 'predictions_log.csv'

predictions_log = pd.read_csv(LOG)
seasons = sorted(predictions_log['season'].unique().tolist())
schedules = nfl.load_schedules(seasons).to_pandas()

done = schedules.dropna(subset=['home_score', 'away_score'])
home_won = (done['home_score'] > done['away_score']).astype(int)
home_won.index = done['game_id']

pending = predictions_log['actual'].isna()
ids = predictions_log.loc[pending, 'game_id']
predictions_log.loc[pending, 'actual'] = ids.map(home_won)
predictions_log.to_csv(LOG, index=False)
 
still = int(predictions_log['actual'].isna().sum())
print(f'Filled {int(pending.sum()) - still} results; {still} still pending')
