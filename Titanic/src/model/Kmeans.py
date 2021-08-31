import statistics

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
import pandas as pd
from Titanic.src.Util.Util import Util, find_optimal_param
from Titanic.src.Util.Util import get_validation_curve

# TODO Adapt file to refactor of Util Class: moving scale_df inside class, refactor of getdf and re optimise for
#  new Features
def main():
    util = Util()
    [X, y, X_test] = util.get_df()
    X.drop('Survived', inplace=True, axis=1)
    X_test.drop('Survived', inplace=True, axis=1)

    print("Columns used: ")
    print(X.columns)

    X = util.scale_df(X)
    X_test = util.scale_df(X_test)
    df = pd.read_csv('../../data/test.csv')

    if util.configValues['tuning']:

        parameter_space = {
            'n_clusters': [2, 4, 10, 50, 100, 150, 200],
            'max_iter': [10, 20, 100, 200, 500, 1000],
        }
        params = {'init': 'random'}
        model = KMeans(**params)
        [params, cv_score, self_score] = find_optimal_param(parameter_space, model, X, y, '../recon/RandomForrest/Kmeans.txt')
        print('Optimal params', params)
        print('Cross Validation score %s' % cv_score)
        print('self Score %s' % str(self_score))

        n_clusters = range(2, 100, 10)
        p = {'n_init': 100, 'max_iter': 1000, 'init': 'random'}
        get_validation_curve(n_clusters, 'n_clusters', KMeans(p), X, y)
        num_clusters = 50
        loss = []
        for i in range(2, 100, 2):
            params = {'n_clusters': i, 'n_init': 100, 'max_iter': 1000, 'init': 'random'}
            model = KMeans(**params)
            model.fit(X)
            loss.append(abs(model.inertia_))
        #     TODO add Axis and figure label here
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(range(2, 100, 2), loss)
        plt.show()

        # print(loss)
        # TODO RE-implement
        # [X, y, x_cv, y_cv] = util.get_df('/train.csv', True, True)
        # X = X[util.configValues['columns']]
        # x_cv = x_cv[util.configValues['columns']]
        # f1 = 0
        # optimal_cluster = 0
        # for i in range(2, 98, 2):
        #     X_norm = scale_df(X)
        #     x_cv_norm = scale_df(x_cv)
        #     print(i)
        #     params = {'n_clusters': i, 'n_init': 100, 'max_iter': 1000, 'init': 'random'}
        #     model = KMeans(**params)
        #     model.fit(X_norm)
        #
        #     X_norm['Cluster'] = model.predict(X_norm)
        #     X_norm['Survived'] = y
        #     mapper = X_norm.groupby('Cluster')['Survived'].mean().round().to_dict()
        #
        #     x_cv_norm['Survived'] = model.predict(x_cv_norm)
        #     try:
        #         overall_predictions = x_cv_norm.Survived.map(mapper).astype(np.int)
        #     except:
        #         print('Ran into exception')
        #     f1_iter = f1_score(y_cv, overall_predictions)
        #     if f1_iter > f1:
        #         f1 = f1_iter
        #         optimal_cluster = i
        #         print("New optimal Cluster size found")
        #         print(optimal_cluster)
        #         print("f1: " + str(f1_iter))


    if util.configValues['predict']:
        params = {'n_clusters': 200, 'n_init': 100, 'max_iter': 200, 'init': 'random'}
        model = KMeans(**params)
        model.fit(X)

        X['Cluster'] = model.predict(X)
        X['Survived'] = y
        mapper = X.groupby('Cluster')['Survived'].mean().round().to_dict()

        df['Survived'] = model.predict(X_test)
        overall_predictions = df.Survived.map(mapper).astype(np.int)

        input_predictions = X['Cluster'].map(mapper).astype(np.int)

        accuracy = accuracy_score(y, input_predictions)

        print("|     accuracy score on train data: %s     |" % str(accuracy))

        print("|    Percentage Survived test set %s       |" % str(overall_predictions.sum()/len(overall_predictions)))

        print("|     Percentage Survived training set %s   |" % str(y.sum()/len(y)))
        print("|     Predicted Percentage Survived training set %s   |" % str(input_predictions.sum()
                                                                              / len(input_predictions)))

        output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': overall_predictions})
        output.to_csv('../../predictions/Kmeans.csv', index=False)
        print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
