import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
from Titanic.src.Util.Util import Util
from Titanic.src.Util.Util import get_validation_curve
from Titanic.src.Util.Util import find_optimal_param


def main():
    util = Util()
    [X, y] = util.get_df('/train.csv')
    y = np.ravel(y)
    X_test = util.get_df('/test.csv', False)
    df = pd.read_csv('../../data/test.csv')

    n_estimators = [100, 300, 500]
    max_depth = [4, 5, 6, 7, 8]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    random_state = [1, 40, 100]

    # Validation curves for parameters
    get_validation_curve(n_estimators, 'n_estimators', RandomForestClassifier(), X, y)
    get_validation_curve(max_depth, 'max_depth', RandomForestClassifier(), X, y)
    get_validation_curve(min_samples_split, 'min_samples_split', RandomForestClassifier(), X, y)
    get_validation_curve(min_samples_leaf, 'min_samples_leaf', RandomForestClassifier(), X, y)
    get_validation_curve(random_state, 'random_state', RandomForestClassifier(), X, y)

    # Find optimal params
    hyper_forrest = {'n_estimators': n_estimators,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'random_state': random_state}
    # # 'criterion': criterion,
    params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 1, 'min_samples_split': 2,
              'min_samples_leaf': 2, 'max_features': 'auto', 'criterion': 'entropy'}
    model = RandomForestClassifier(**params)
    find_optimal_param(hyper_forrest, model, X, y, 'RandForrest.txt')

    # Use optimal Model
    params = {'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100,
              'random_state': 1, 'max_features': 'auto', 'criterion': 'entropy'}
    model = RandomForestClassifier(**params)
    overall_predictions = {}
    [X, y, x_cv, y_cv] = util.get_df('/train.csv', True, True)
    X.drop('Survived', inplace=True, axis=1)
    x_cv.drop('Survived', inplace=True, axis=1)
    f1 = 0
    for i in range(10):
        model = RandomForestClassifier(**params)
        y = np.ravel(y)
        model.fit(X, y)
        predictions = model.predict(x_cv)
        f1_iter = f1_score(y_cv, predictions)
        if f1_iter > f1:
            overall_predictions = model.predict(X_test)
            f1 = f1_iter
    output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': overall_predictions})
    output.to_csv('RandomForrestOptimized.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
