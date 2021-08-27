import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
from Titanic.src.Util.Util import Util
from Titanic.src.Util.Util import get_validation_curve
from Titanic.src.Util.Util import find_optimal_param
from Titanic.src.Util.Util import scale_df

# TODO Adapt file to refactor of Util Class: moving scale_df inside class, refactor of getdf and re optimise for
#  new Features
def main():
    util = Util()
    [X, y] = util.get_df('/train.csv')
    X = X[util.configValues['columns']]
    X =scale_df(X)
    y = np.ravel(y)
    X_test = util.get_df('/test.csv', False)
    X_test = X_test[util.configValues['columns']]
    X_test = scale_df(X_test)
    df = pd.read_csv('../../data/test.csv')

    n_estimators = [100, 150, 300, 500]
    max_depth = [4, 5, 6, 7, 8]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    random_state = [1, 40, 100]

    if util.configValues['tuning']:
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

        params = {'max_features': 'auto', 'criterion': 'entropy'}
        model = RandomForestClassifier(**params)
        optimap_params = find_optimal_param(hyper_forrest, model, X, y, '../recon/RandomForrest/RandForrest.txt')
        print(optimap_params)

    if util.configValues['predict']:

        # Use optimal Model
        params = {'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100,
                  'random_state': 100, 'max_features': 'auto', 'criterion': 'entropy'}

        overall_predictions = {}
        # [X, y, x_cv, y_cv] = util.get_df('/train.csv', True, True)
        # X = X[util.configValues['columns']]
        # x_cv = x_cv[util.configValues['columns']]
        #
        # X = scale_df(X)
        # # x_cv = scale_df(x_cv)

        # X.drop('Survived', inplace=True, axis=1)
        # x_cv.drop('Survived', inplace=True, axis=1)
        f1 = 0
        for i in range(30):
            model = RandomForestClassifier(**params)
            y = np.ravel(y)
            model.fit(X, y)
            predictions = model.predict(X)
            f1_iter = f1_score(y, predictions)
            if f1_iter > f1:
                overall_predictions = model.predict(X_test)
                f1 = f1_iter
        print("f1 score on train data: %s" % str(f1))
        print(overall_predictions.sum()/len(overall_predictions))
        print(y.sum()/len(y))
        output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': overall_predictions})
        output.to_csv('../../predictions/RandomForrestOptimized.csv', index=False)
        print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
