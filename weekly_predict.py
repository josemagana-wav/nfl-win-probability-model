import os
from datetime import datetime, timezone
import pandas as pd
from src.data import load_data, clean_plays, current_season
from src.features import build_upcoming_features
from src.model import FEATURE_COLS, load_model

LOG = 'predictions_log.csv'
SEASON = current_season()

schedules, pbp= load_data([SEASON - 1, SEASON])
plays = clean_plays(pbp)

unplayed = schedules[(schedules['season'] == SEASON) &
schedules['home_score'].isna()]
if unplayed.empty:
    raise SystemExit(f'No unplayed {SEASON} games found (season over?).')

week = unplayed['week'].min()
upcoming = unplayed[unplayed['week'] == week]

up = build_upcoming_features(plays, upcoming)
missing = up[up[FEATURE_COLS].isna().any(axis=1)]
if len(missing):
    print ('Skipping games with missing features:', list(missing['game_id']))
up = up.dropna(subset=FEATURE_COLS).copy()

COLUMNS = ['game_id', 'season', 'week', 'home_team', 'away_team', 'spread_line',
           'pred_prob_home_win', 'predicted_label', 'logged_at', 'actual']

model = load_model()
up['pred_prob_home_win'] = model.predict_proba(up[FEATURE_COLS])[:, 1]
up['predicted_label'] = (up['pred_prob_home_win'] > 0.5).astype(int)
up['logged_at'] = datetime.now(timezone.utc).isoformat(timespec = 'seconds')
up['actual'] = pd.NA
new_rows = up[COLUMNS]

if os.path.exists(LOG):
    predictions_log = pd.read_csv(LOG)
    new_rows = new_rows[~new_rows['game_id'].isin(predictions_log['game_id'])]
    predictions_log = pd.concat([predictions_log, new_rows], ignore_index=True)
else:
    predictions_log = new_rows

predictions_log.to_csv(LOG, index=False)
print(f'Logged {len(new_rows)} new games for {SEASON} week {week}')