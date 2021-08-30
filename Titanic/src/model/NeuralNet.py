import statistics

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
from Titanic.src.Util.Util import Util
from Titanic.src.Util.Util import find_optimal_param
from sklearn.model_selection import cross_val_score

def main():
    util = Util()
    [X, y, X_test] = util.get_df()
    X.drop('Survived', inplace=True, axis=1)
    X_test.drop('Survived', inplace=True, axis=1)

    print("Columns used: ")
    print(X.columns)

    X = util.scale_df(X)
    X_test = util.scale_df(X_test)

    y = np.ravel(y)

    df = pd.read_csv('../../data/test.csv')

    if util.configValues['tuning']:
        parameter_space = {
            'hidden_layer_sizes': [(17, 3), (16, 3), (16, 16, 16), (16,), (16, 16, 16, 3)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': [0.0001, 0.02],
            'learning_rate': ['constant'],
            'max_iter': [10, 20, 50, 100, 300, 500, 1000, 1500]
        }
        mlp = MLPClassifier(random_state=1)
        [params, cv_score, self_score] = find_optimal_param(parameter_space, mlp, X, y, '../recon/RandomForrest/MLPIter.txt')
        print('Optimal params', params)
        print('Cross Validation score %s' % cv_score)
        print('self Score %s' % str(self_score))

        global_min = 0
        gmin_param = []
        gmean_param = []
        global_mean = 0

        max_iter = [5, 10, 20, 30, 40, 50, 100]
        # 'hidden_layer_sizes'=[(17, 3), (17, 2), (17, 4)],
        for i in max_iter:
            print('Evaluating iter: %s' % str(i))
            params = {'activation': 'tanh', 'alpha': 0.02, 'hidden_layer_sizes': (17, ), 'learning_rate': 'constant',
                      'max_iter': i, 'solver': 'lbfgs', 'random_state': 1, 'verbose': 0}
            mlp = MLPClassifier(**params)
            cv_scores_mean = []
            cv_scores_min = []
            for j in range(50):
                cv = cross_val_score(mlp, X, y)
                cv_scores_mean.append(statistics.mean(cv))
                cv_scores_min.append(min(cv))

            cv_score_mean = statistics.mean(cv_scores_mean)
            cv_score_min = min(cv_scores_min)

            if cv_score_min > global_min:
                print('New local min found at ', params, file=open('../recon/ManualOptimization', 'a'))
                print('Min score ', cv_score_min, file=open('../recon/ManualOptimization', 'a'))
                global_min = cv_score_min
                gmin_param = params

            if cv_score_mean > global_mean:
                print('New local mean found at ', params, file=open('../recon/ManualOptimization', 'a'))
                print('Mean score ', cv_score_mean, file=open('../recon/ManualOptimization', 'a'))
                global_mean = cv_score_mean
                gmean_param = params

        print('Global min found at ', gmin_param, file=open('../recon/ManualOptimization', 'a'))
        print('Global min ', global_min, file=open('../recon/ManualOptimization', 'a'))
        print('Global mean found at ', gmean_param, file=open('../recon/ManualOptimization', 'a'))
        print('Global mean ', global_mean, file=open('../recon/ManualOptimization', 'a'))

    if util.configValues['predict']:
        params = {'activation': 'tanh', 'alpha': 0.02, 'hidden_layer_sizes': (17,), 'learning_rate': 'constant',
                  'max_iter': 20, 'solver': 'lbfgs', 'random_state': 1}
        mlp = MLPClassifier(**params)
        mlp.fit(X, y)
        overall_predictions = mlp.predict(X_test)
        predictions_train = mlp.predict(X).astype(int)
        f1 = f1_score(y, predictions_train)
        accuracy = accuracy_score(y,predictions_train)

        # Evaluating model as a sanity check before submitting
        cv_scores_mean = []
        cv_scores_min = []
        for i in range(50):
            cv = cross_val_score(mlp, X, y)
            cv_scores_mean.append(statistics.mean(cv))
            cv_scores_min.append(min(cv))

        cv_score_mean = statistics.mean(cv_scores_mean)
        cv_score_min = min(cv_scores_min)

        print("-----------------------------------------------")
        print("|        f1 score on train data: %s        |" % str(f1))
        print("|     accuracy score on train data: %s     |" % str(accuracy))

        print("|  cv score mean %s |" % cv_score_mean)
        print("|  cv score min %s |" % cv_score_min)

        print("|    Percentage Survived test set %s       |" % str(overall_predictions.sum()/len(overall_predictions)))

        print("|     Percentage Survived training set %s   |" % str(y.sum()/len(y)))

        print("|Percentage predicted Survived training set %s |" % str(predictions_train.sum()/len(predictions_train)))

        print("|        Neural net mean accuracy %s      |" % str(mlp.score(X, y)))
        print("-----------------------------------------------")

        output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': overall_predictions})
        output.to_csv('../../predictions/NeuralNet.csv', index=False, float_format='%.0f')
        print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
