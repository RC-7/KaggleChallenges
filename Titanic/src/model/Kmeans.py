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


def main():
    util = Util()
    [X, y] = util.get_df('/train.csv')
    y = np.ravel(y)
    X_test = util.get_df('/test.csv', False)
    df = pd.read_csv('../../data/test.csv')
    X_norm = scale_df(X)

    X_test_norm = scale_df(X_test)
    positive_values = np.where(y == 1)
    # Look at loss for nr clusters

    if util.configValues['tuning']:
        num_clusters = 50
        loss = []
        for i in range(2, 100, 2):
            params = {'n_clusters': i, 'n_init': 100, 'max_iter': 1000, 'init': 'random'}
            model = KMeans(**params)
            model.fit(X_norm)
            loss.append(model.inertia_)
        #     TODO add Axis and figure label here
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(range(2, 100, 2), loss)
        plt.show()

        [X, y, x_cv, y_cv] = util.get_df('/train.csv', True, True)
        f1 = 0
        optimal_cluster = 0
        for i in range(2, 98, 2):
            X_norm = scale_df(X)
            x_cv_norm = scale_df(x_cv)
            print(i)
            params = {'n_clusters': i, 'n_init': 100, 'max_iter': 1000, 'init': 'random'}
            model = KMeans(**params)
            model.fit(X_norm)

            X_norm['Cluster'] = model.predict(X_norm)
            X_norm['Survived'] = y
            mapper = X_norm.groupby('Cluster')['Survived'].mean().round().to_dict()

            x_cv_norm['Survived'] = model.predict(x_cv_norm)
            try:
                overall_predictions = x_cv_norm.Survived.map(mapper).astype(np.int)
            except:
                print('Ran into exception')
            f1_iter = f1_score(y_cv, overall_predictions)
            if f1_iter > f1:
                f1 = f1_iter
                optimal_cluster = i
                print("New optimal Cluster size found")
                print(optimal_cluster)
                print("f1: " + str(f1_iter))

        print(loss)
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
