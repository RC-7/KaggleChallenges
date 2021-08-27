import itertools
import statistics

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn import preprocessing
import pandas as pd
from Titanic.src.Util.Util import Util
from Titanic.src.Util.Util import get_validation_curve
from Titanic.src.Util.Util import scale_df
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    util = Util()
    [X, y] = util.get_df('/train.csv')
    X = X[util.configValues['columns']]
    y = np.ravel(y)
    X_test = util.get_df('/test.csv', False)
    X_test = X_test[util.configValues['columns']]
    df = pd.read_csv('../../data/test.csv')
    X_norm = scale_df(X)

    X_test_norm = scale_df(X_test)
    positive_values = np.where(y == 1)
    # Look at loss for nr clusters
    # for L in range(0, len(util.configValues['allColumns']) + 1):
    #     for subset in itertools.combinations(util.configValues['allColumns'], L):
    #         if len(subset) == 0:
    #             print(subset)

    if util.configValues['tuning']:
        # num_clusters = 50
        # loss = []
        # for i in range(2, 100, 2):
        #     params = {'n_clusters': i, 'n_init': 100, 'max_iter': 1000, 'init': 'random'}
        #     model = KMeans(**params)
        #     model.fit(X_norm)
        #     loss.append(model.inertia_)
        # #     TODO add Axis and figure label here
        # fig = plt.figure()
        # ax = plt.axes()
        # ax.plot(range(2, 100, 2), loss)
        # plt.show()
        # print(loss)
        stuff = [1, 2, 3]
        globalOptima = []
        globalOptimaf1 = 0
        [X_template, y_template, x_cv_template, y_cv_template] = util.get_df('/train.csv', True, True)
        for L in range(0, len(util.configValues['columns']) + 1):
            for subset in itertools.combinations(util.configValues['columns'], L):
                if len(subset) == 5:
                    break
                subset = np.asarray(subset)
                # print(np.asarray(subset))
                # print(type(subset))
                print('---------------------', file=open('kmeansFeatureBruteForce.txt', 'a'))
                print('Optimising feature Set', subset, file=open('kmeansFeatureBruteForce.txt', 'a'))

                X = X_template
                y = y_template
                x_cv = x_cv_template
                y_cv = y_cv_template
                X = X[util.configValues['columns']]
                x_cv =  x_cv[util.configValues['columns']]
                # [X, y, x_cv, y_cv] = util.get_df('/train.csv', True, True)
                X_norm = scale_df(X)
                x_cv_norm = scale_df(x_cv)
                f1 = 0
                optimal_cluster = 0
                for i in range(20, 40, 10):
                    # print(i)
                    X_norm = scale_df(X)
                    x_cv_norm = scale_df(x_cv)
                    params = {'n_clusters': i, 'n_init': 100, 'max_iter': 400, 'init': 'random'}
                    model = KMeans(**params)
                    model.fit(X_norm)

                    X_norm['Cluster'] = model.predict(X_norm)
                    X_norm['Survived'] = y
                    mapper = X_norm.groupby('Cluster')['Survived'].mean().round().to_dict()

                    x_cv_norm['Survived'] = model.predict(x_cv_norm)
                    try:
                        overall_predictions = x_cv_norm.Survived.map(mapper).astype(np.int)
                        f1_iter = f1_score(y_cv, overall_predictions)
                        if f1_iter > f1:
                            f1 = f1_iter
                            optimal_cluster = i
                            if f1 > globalOptimaf1:
                                globalOptimaf1 = f1
                                globalOptima = subset
                    except:
                        # print('Ran into exception')
                        pass

                        # print("New optimal Cluster size found")
                        # print(optimal_cluster)
                        # print("f1: " + str(f1_iter))
                print('Optimal clusters %s with f1 score %s' % (optimal_cluster, f1),
                      file=open('kmeansFeatureBruteForce.txt', 'a'))
                print(globalOptima, file=open('kmeansFeatureBruteForce.txt', 'a'))
                print(globalOptimaf1, file=open('kmeansFeatureBruteForce.txt', 'a'))
                print('GlobalOptima:', globalOptima,
                      file=open('kmeansFeatureBruteForce.txt', 'a'))
                print('GlobalOptima %s' % str(globalOptimaf1),
                      file=open('kmeansFeatureBruteForce.txt', 'a'))

    if util.configValues['predict']:
        params = {'n_clusters': 60, 'n_init': 100, 'max_iter': 1000, 'init': 'random'}
        model = KMeans(**params)
        model.fit(X_norm)

        X_norm['Cluster'] = model.predict(X_norm)
        X_norm['Survived'] = y
        mapper = X_norm.groupby('Cluster')['Survived'].mean().round().to_dict()

        df['Survived'] = model.predict(X_test_norm)
        overall_predictions = df.Survived.map(mapper).astype(np.int)

        output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': overall_predictions})
        output.to_csv('KmeansNormalized.csv', index=False)
        print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
