from src.data import SEASONS, load_data, clean_plays, build_target
from src.features import build_features
from src.model import split_by_season, train_model, coef_inspection, save_model

schedules, pbp = load_data()
games = build_features(clean_plays(pbp), build_target(schedules))

X_train, y_train, _, _ = split_by_season(games, test_season=max(SEASONS) + 1)
model = train_model(X_train, y_train)
coef_inspection(model)
save_model(model)
print(f'Trained on {len(X_train)} games ({min(SEASONS)}-{max(SEASONS)}); saved model.pkl')