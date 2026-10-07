from src.data import load_data, clean_plays, build_target, current_season
from src.features import build_features
from src.model import split_by_season, train_model
from src.evaluate import log_predictions

SEASON = current_season()

schedules, pbp = load_data()
plays = clean_plays(pbp)
games = build_target(schedules)
games = build_features(plays, games)

# Each run trains only on seasons before the one it predicts, so every
# prediction is out-of-sample.
RUNS = [
    (SEASON - 1, 'backtest_log.csv'),  # last full season
    (SEASON, 'backfill_log.csv'),      # current-season games already played
]
for test_season, path in RUNS:
    X_train, y_train, X_test, y_test = split_by_season(games, test_season=test_season)
    if X_test.empty:
        print(f'No finished {test_season} games yet; skipped {path}')
        continue
    model = train_model(X_train, y_train)
    log = log_predictions(model, X_test, y_test, games, path=path)
    print(f'Wrote {len(log)} {test_season} predictions to {path}')
