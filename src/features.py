#%%
def _team_form(plays, team_col, out_name, window=8, min_periods=3):
    tg = (
        plays.groupby(['season', 'week', team_col])['epa']
        .mean()
        .reset_index()
        .rename(columns={team_col: 'team', 'epa': 'game_epa'})
        .sort_values(['team', 'season', 'week'])
    )

    tg[out_name] = (
        tg.groupby('team')['game_epa']
        .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    )

    return tg[['season', 'week', 'team', out_name]]
#%%
def build_features(plays, games):
    off = _team_form(plays, 'posteam', 'form_off_epa')
    dfn = _team_form(plays, 'defteam', 'form_def_epa')
    form = off.merge(dfn, on=['season', 'week', 'team'])

    team_game_home = form.rename(columns={
        'team' : 'home_team',
        'form_off_epa' : 'home_form_off_epa',
        'form_def_epa' : 'home_form_def_epa'
    })
    team_game_away = form.rename(columns={
        'team' : 'away_team',
        'form_off_epa' : 'away_form_off_epa',
        'form_def_epa' : 'away_form_def_epa'
    })

    games = games.merge(team_game_home, on=['season', 'week', 'home_team'])
    games = games.merge(team_game_away, on=['season', 'week', 'away_team'])



    games['off_epa_diff'] = games['home_form_off_epa'] - games['away_form_off_epa']
    games['def_epa_diff'] = games['home_form_def_epa'] - games['away_form_def_epa']

    games['home_field'] = (games['location'] == 'Home').astype(int)
    games['rest_diff'] = games['home_rest'] - games['away_rest']
    games['div_game'] = games['div_game'].astype(int)
    return games
# %%
def _latest_form(plays, team_col, out_name, window=8):
    tg = (
        plays.groupby(['season', 'week', team_col])['epa']
        .mean()
        .reset_index()
        .rename(columns={team_col: 'team', 'epa': 'game_epa'})
        .sort_values(['team', 'season', 'week'])
    )
    return (
        tg.groupby('team')['game_epa']
        .apply(lambda s : s.tail(window).mean())
        .rename(out_name)
    )

def build_upcoming_features(plays, upcoming):
    off = _latest_form(plays, 'posteam', 'form_off_epa')
    dfn = _latest_form(plays, 'defteam', 'form_def_epa')

    up = upcoming.copy()
    up['off_epa_diff'] = up['home_team'].map(off) - up['away_team'].map(off)
    up['def_epa_diff'] = up['home_team'].map(dfn) - up['away_team'].map(dfn)
    up['home_field'] = (up['location'] == 'Home').astype(int)
    up['rest_diff'] = up['home_rest'] - up['away_rest']
    up['div_game'] = up['div_game'].astype(int)
    return up