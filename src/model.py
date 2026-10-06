
feature_cols = ['off_epa_diff', 'def_epa_diff', 'home_field', 'rest_diff', 'div_game']

def split_by_season(games, feature_cols, test_season= 2025):
    games = games.dropna(subset=feature_cols)

    train = games[games['season'] < test_season]
    test = games[games['season'] == test_season]

    return(train[feature_cols], train['home_win'],
           test[feature_cols], test['home_win'])