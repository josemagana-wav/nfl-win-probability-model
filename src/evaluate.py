# src/evaluate.py
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, log_loss)
from sklearn.calibration import calibration_curve


def evaluate(model, X_test, y_test, games, n_bins=5):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_prob))
    print('Log loss:', log_loss(y_test, y_prob))

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual win rate')
    plt.legend()
    plt.show()

    # Skip games with no line or a pick'em (spread 0): Vegas has no favorite there
    test = games.loc[X_test.index].dropna(subset=['spread_line'])
    test = test[test['spread_line'] != 0].copy()

    test['vegas_pred'] = (test['spread_line'] > 0).astype(int)
    print('Vegas-implied accuracy:',
          accuracy_score(test['home_win'], test['vegas_pred']))


def log_predictions(model, X_test, y_test, games, path='backtest_log.csv'):
    log = games.loc[X_test.index, ['game_id', 'season', 'week', 'home_team',
                                   'away_team', 'spread_line']].copy()
    log['pred_prob_home_win'] = model.predict_proba(X_test)[:, 1]
    log['predicted_label'] = model.predict(X_test)
    log['actual'] = y_test
    log.sort_values('week').to_csv(path, index=False)
    return log
