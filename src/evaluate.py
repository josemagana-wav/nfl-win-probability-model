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

    test = games.loc[X_test.index].dropna(subset=['spread_line']).copy()

    test['vegas_pred'] = (test['spread_line'] > 0).astype(int)
    print('Vegas-implied accuracy:',
          accuracy_score(test['home_win'], test['vegas_pred']))