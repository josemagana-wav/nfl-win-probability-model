import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title='NFL Win Probability Tracker', layout='wide')
 
LIVE = 'predictions_log.csv'
BACKFILL = 'backfill_log.csv'  # current-season games played before live logging began
BACKTEST = 'backtest_log.csv'

 
@st.cache_data(ttl=300)
def load_log(path):
    df = pd.read_csv(path)
    df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
 
    home_pick = df['predicted_label'] == 1
    p_home = df['pred_prob_home_win']
    df['pick'] = df['home_team'].where(home_pick, df['away_team'])
    df['confidence'] = p_home.where(home_pick, 1 - p_home)
    df['matchup'] = df['away_team'] + ' @ ' + df['home_team']
 
    finished = df['actual'].notna()
    right = df['predicted_label'] == df['actual']
    df['result'] = 'Pending'
    df.loc[finished & right, 'result'] = 'Correct'
    df.loc[finished & ~right, 'result'] = 'Wrong'
 
    df['winner'] = ''
    df.loc[df['actual'] == 1, 'winner'] = df['home_team']
    df.loc[df['actual'] == 0, 'winner'] = df['away_team']
 
    # Vegas benchmark: a positive spread_line means the home team is favored
    if 'spread_line' in df.columns:
        vegas_label = (df['spread_line'] > 0).astype(int)
        # No line or a pick'em (spread 0) means no Vegas favorite, so leave it out
        has_fav = df['spread_line'].notna() & (df['spread_line'] != 0)
        df['vegas_correct'] = (vegas_label == df['actual']).where(finished & has_fav)
    else:
        df['vegas_correct'] = pd.NA
    return df


def load_live(include_backfill):
    parts = []
    if os.path.exists(LIVE):
        parts.append(load_log(LIVE).assign(logged='Live'))
    if include_backfill:
        backfill = load_log(BACKFILL).assign(logged='Backfill')
        # A game in both logs keeps its live (pre-kickoff) prediction
        if parts:
            backfill = backfill[~backfill['game_id'].isin(parts[0]['game_id'])]
        parts.append(backfill)
    return pd.concat(parts, ignore_index=True) if parts else None


# ---------- sidebar: choose data and filters ----------
available = []
if os.path.exists(LIVE) or os.path.exists(BACKFILL):
    available.append('Live tracker')
if os.path.exists(BACKTEST):
    available.append('Backtest (last full season)')
st.title('NFL Win Probability Tracker')

if not available:
    st.warning('No prediction logs found yet. Run `python weekly_predict.py` '
               '(live) or `python backtest.py` (backtest) first.')
    st.stop()

st.sidebar.header('Filters')
choice = st.sidebar.radio('Data', available)
if choice == 'Live tracker':
    include_backfill = os.path.exists(BACKFILL) and st.sidebar.checkbox(
        'Include backfilled games', value=True,
        help='Games played before live tracking began, predicted afterwards '
             'by a model trained only on earlier seasons.')
    log = load_live(include_backfill)
else:
    log = load_log(BACKTEST).assign(logged='Backtest')

if log is None or log.empty:
    st.info('No games to show. Tick "Include backfilled games" or run '
            '`python weekly_predict.py`.')
    st.stop()

weeks = sorted(int(w) for w in log['week'].unique())
if len(weeks) > 1:
    first, last = st.sidebar.select_slider(
        'Weeks', options=weeks, value=(weeks[0], weeks[-1]))
else:
    first = last = weeks[0]
show = st.sidebar.radio('Show', ['All games', 'Finished only', 'Pending only'])
 
view = log[(log['week'] >= first) & (log['week'] <= last)]
if show == 'Finished only':
    view = view[view['result'] != 'Pending']
elif show == 'Pending only':
    view = view[view['result'] == 'Pending']
 
if choice.startswith('Backtest'):
    st.caption('Backtest: predictions for games already played, made by a model '
               'trained only on earlier seasons. A fair out-of-sample test, '
               'but not live tracking.')
elif (log['logged'] == 'Backfill').any():
    st.caption('Live: predictions marked "Live" were logged before kickoff. '
               '"Backfill" games were played before tracking began and were predicted '
               'afterwards by a model trained only on earlier seasons. '
               'Player injuries are not taken into account.')
else:
    st.caption('Live: every prediction was logged before kickoff; '
               'results are filled in afterwards. Player injuries '
               'were not taken into account')
 
# ---------- headline numbers ----------
finished = view[view['result'] != 'Pending']
c1, c2, c3, c4 = st.columns(4)
if finished.empty:
    c1.metric('Model accuracy', 'n/a')
    c2.metric('Record', '0-0')
    c3.metric('Vegas accuracy', 'n/a')
else:
    right = int((finished['result'] == 'Correct').sum())
    wrong = int((finished['result'] == 'Wrong').sum())
    c1.metric('Model accuracy', f'{right / (right + wrong):.1%}')
    c2.metric('Record (right-wrong)', f'{right}-{wrong}')
    vegas = finished['vegas_correct'].dropna().astype(float)
    c3.metric('Vegas accuracy', f'{vegas.mean():.1%}' if len(vegas) else 'n/a')
c4.metric('Games pending', int((view['result'] == 'Pending').sum()))
 
tab_games, tab_trend, tab_conf, tab_about = st.tabs(
    ['Games', 'Accuracy over time', 'Confidence check', 'About'])
 
# ---------- tab 1: the games ----------
with tab_games:
    table = (
        view.sort_values(['week', 'confidence'], ascending=[True, False])
            .assign(confidence_pct=lambda d: (d['confidence'] * 100).round(0))
            [['week', 'matchup', 'pick', 'confidence_pct', 'result', 'winner', 'logged']]
    )
    if table.empty:
        st.info('No games match these filters.')
    else:
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                'week': 'Week',
                'matchup': 'Matchup',
                'pick': 'Model pick',
                'confidence_pct': st.column_config.ProgressColumn(
                    'Confidence (%)', min_value=50, max_value=100, format='%.0f'),
                'result': 'Result',
                'winner': 'Actual winner',
                'logged': 'Logged',
            },
        )
 
# ---------- tab 2: accuracy over time ----------
with tab_trend:
    done = view[view['result'] != 'Pending']
    if done.empty:
        st.info('No finished games yet. Results appear after '
                '`python weekly_update.py` runs.')
    else:
        weekly = (
            done.assign(correct=done['result'] == 'Correct')
                .groupby('week')['correct'].agg(['sum', 'count'])
                .sort_index()
        )
        chart = pd.DataFrame({
            'Weekly accuracy': weekly['sum'] / weekly['count'],
            'Season-to-date accuracy': weekly['sum'].cumsum() / weekly['count'].cumsum(),
        })
        st.line_chart(chart)
        st.caption('A single week is only about 15 games, so the weekly line is '
                   'noisy. Watch the season-to-date line.')
 
# ---------- tab 3: does confidence mean anything? ----------
with tab_conf:
    done = view[view['result'] != 'Pending']
    if done.empty:
        st.info('No finished games yet.')
    else:
        buckets = pd.cut(done['confidence'], bins=[0.5, 0.55, 0.6, 0.7, 1.0],
                         labels=['50-55%', '55-60%', '60-70%', '70%+'],
                         include_lowest=True)
        by_conf = (
            done.assign(correct=done['result'] == 'Correct', bucket=buckets)
                .groupby('bucket', observed=True)['correct']
                .agg(['mean', 'count'])
        )
        by_conf.index = by_conf.index.astype(str)
        st.bar_chart(by_conf['mean'].rename('Accuracy'))
        st.dataframe(by_conf.rename(columns={'mean': 'Accuracy', 'count': 'Games'}))
        st.caption('If the model is well calibrated, accuracy should rise from left '
                   'to right. Buckets with few games are noisy.')
 
# ---------- tab 4: about ----------
with tab_about:
    st.markdown(
        '''
**What this is:** a logistic regression trained on rolling offensive and
defensive EPA/play, home field, rest days and divisional games.
 
**Confidence** is the probability the model gives to its own pick (always 50%
or higher).
 
**Vegas accuracy** is how often the closing spread's favorite won, the
benchmark to beat.
 
**Limits:** the model does not know about injuries or quarterback changes.
'''
    )
