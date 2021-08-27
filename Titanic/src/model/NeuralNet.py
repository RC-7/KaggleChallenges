import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import pandas as pd
from Titanic.src.Util.Util import Util
from Titanic.src.Util.Util import get_validation_curve
from Titanic.src.Util.Util import find_optimal_param
from sklearn.model_selection import cross_val_score


def main():
    util = Util()
    [X, y] = util.get_df('/train.csv')
    X = X[util.configValues['columns']]
    X = util.scale_df(X)
    y = np.ravel(y)
    X_test = util.get_df('/test.csv', False)
    X_test = X_test[util.configValues['columns']]
    X_test = util.scale_df(X_test)
    df = pd.read_csv('../../data/test.csv')

    if util.configValues['tuning']:
        parameter_space = {
            'hidden_layer_sizes': [(50,50), (5,3), (50, 50, 50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
        mlp = MLPClassifier(max_iter=10000)
        [params, cv_score, self_score] = find_optimal_param(parameter_space, mlp, X, y, '../recon/RandomForrest/MLP.txt')
        print('Optimal params', params)
        print('Cross Validation score %s' % cv_score)
        print('self Score %s' % str(self_score))

    if util.configValues['predict']:
        alpha = 1e-5
        #Other perams learning_rate, batch_size.
        params = {"solver": 'adam', 'alpha': alpha, 'hidden_layer_sizes': (50, 50), 'random_state': 1,
                  "max_iter": 10000, 'activation': 'tanh', 'learning_rate': 'adaptive'}
        mlp = MLPClassifier(**params)
        mlp.fit(X, y)
        overall_predictions = mlp.predict(X_test)
        predictions_train = mlp.predict(X)
        f1 = f1_score(y, predictions_train)

        cv_score =cross_val_score(mlp, X, y)

        print("f1 score on train data: %s" % str(f1))

        print("cv score %s" % cv_score)

        print(overall_predictions.sum()/len(overall_predictions))

        print(y.sum()/len(y))

        print(predictions_train.sum()/len(predictions_train))

        print(mlp.score(X,y))

        output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': overall_predictions})
        output.to_csv('../../predictions/NeuralNet.csv', index=False)
        print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
