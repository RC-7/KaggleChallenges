import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from Titanic.src.Util.Util import Util
from Titanic.src.Util.Util import get_validation_curve
from Titanic.src.Util.Util import find_optimal_param

# TODO Adapt file to refactor of Util Class: moving scale_df inside class, refactor of getdf and re optimise for
#  new Features
def main():
    util = Util()
    [X, y, X_test] = util.get_df()
    X.drop('Survived', inplace=True, axis=1)
    X_test.drop('Survived', inplace=True, axis=1)

    print("Columns used: ")
    print(X.columns)

    y = np.ravel(y)
    X = util.scale_df(X)
    X_test = util.scale_df(X_test)
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
        params = {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150,
                  'random_state': 40, 'max_features': 'auto', 'criterion': 'entropy'}

        overall_predictions = {}
        f1 = 0
        for i in range(30):
            model = RandomForestClassifier(**params)
            y = np.ravel(y)
            model.fit(X, y)
            predictions = model.predict(X)
            f1_iter = f1_score(y, predictions)
            if f1_iter > f1:
                overall_predictions = model.predict(X_test).astype(int)
                f1 = f1_iter
        print("f1 score on train data: %s" % str(f1))
        input_predictions = model.predict(X).astype(int)
        accuracy = accuracy_score(y, input_predictions)

        print("|     accuracy score on train data: %s     |" % str(accuracy))
        print(overall_predictions.sum()/len(overall_predictions))
        print("|     Percentage Survived training set %s   |" % str(y.sum() / len(y)))
        print("|     Predicted Percentage Survived training set %s   |" % str(input_predictions.sum()
                                                                             / len(input_predictions)))
        print(
            "|    Percentage Survived test set %s       |" % str(overall_predictions.sum() / len(overall_predictions)))
        output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': overall_predictions})
        output.to_csv('../../predictions/RandomForrestOptimized.csv', index=False)
        print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
