from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

FEATURE_COLS = ['off_epa_diff', 'def_epa_diff', 'home_field', 'rest_diff', 'div_game']

def split_by_season(games, feature_cols=FEATURE_COLS, test_season= 2025):
    games = games.dropna(subset=feature_cols)

    train = games[games['season'] < test_season]
    test = games[games['season'] == test_season]

    return(train[feature_cols], train['home_win'],
           test[feature_cols], test['home_win'])

def train_model(X_train, y_train, C=1.0):
    model = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=C, max_iter=1000)),])
    model.fit(X_train, y_train)
    return model

def coef_inspection(model, feature_cols=FEATURE_COLS):
    coefs = pd.Series(model.named_steps['clf'].coef_[0], index=feature_cols)
    print(coefs.sort_values())