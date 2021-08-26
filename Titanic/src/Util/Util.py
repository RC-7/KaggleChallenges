import json
import numpy as np
import pandas as pd

# import tensorflow as tf
import sklearn.model_selection
from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve, GridSearchCV


def group_by_sub_string(original_string, list_of_sub_strings):
    for substring in list_of_sub_strings:
        if original_string.find(substring) != -1:
            return substring
    return np.nan


def get_validation_curve(param_array, param_name, classifier, X, y):
    param_range = param_array
    train_scores, test_scores = validation_curve(
        classifier,
        X=X, y=y,
        param_name=param_name,
        param_range=param_array, cv=3)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with Random Forest" + "(" + param_name + ")")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


def find_optimal_param(hyper_peram, model, X, y, output_file):
    grid_f = GridSearchCV(estimator=model, param_grid=hyper_peram, cv=3, verbose=1,
                          n_jobs=-1)
    print(grid_f.fit(X, y), file=open(output_file, 'a'))
    print("params ", file=open(output_file, 'a'))
    print(grid_f.best_params_, file=open(output_file, 'a'))
    print("estimator ", file=open(output_file, 'a'))
    print(grid_f.best_estimator_, file=open(output_file, 'a'))
    print("index", file=open(output_file, 'a'))
    print(grid_f.best_index_, file=open(output_file, 'a'))
    print("score", file=open(output_file, 'a'))
    print(grid_f.best_score_, file=open(output_file, 'a'))
    return grid_f.best_params_

class Util:
    def __init__(self):
        self.configValues = {}
        try:
            config_file = open("../config.JSON")
            self.configValues = json.load(config_file)
        except (OSError, FileNotFoundError) as e:
            self.configValues["env"] = "Kaggle"
        self.dataPath = "../../data" if self.configValues["env"] == "Local" else "/kaggle/input"

    def normalise(self, dataframe):
        pass

    # def convert_to__normalised_tensor(self, df):
    #     # tf.convert_to_tensor(df)
    #     normalizer = tf.keras.layers.LayerNormalization(axis=-1)
    #     # normalizer = tf.keras.layers.Normalization(axis=-1)
    #     print(df)
    #     print(normalizer(df))
    #     # normalizer.adapt(df)
    #     # print(df.iloc[:3])
    #     # print(normalizer(df.iloc[:3]))

    def get_df(self, dataset_name, train_set=True, get_cv=False):
        df = pd.read_csv(self.dataPath + dataset_name)
        # Seperate only surnames from Passenger Names
        df['Surname'] = df['Name'].map(lambda x: x.split(',')[0])
        surname_count = df.groupby('Surname').size()
        df['SurnameCount'] = df['Surname'].map(lambda x: surname_count[x])
        # Drop Surname column
        df.drop('Surname', inplace=True, axis=1)
        # Drop Name column
        df.drop('Name', inplace=True, axis=1)

        # Drop ID column
        df.drop('PassengerId', inplace=True, axis=1)
        # Binarise Sex
        df['Sex'].replace({"female": 1, "male": 0}, inplace=True)
        # Convert embarked to Numbers
        df['Embarked'].replace({"S": 3, "C": 1, "Q": 0}, inplace=True)
        # Raplace Nan values, Will try to do this a bit more intelligently
        df["Embarked"] = df["Embarked"].fillna("3")
        df["Cabin"] = df["Cabin"].fillna("Unknown")

        # Fix this
        # val = df.Fare.where(df.Pclass == 3).mean()
        # df.Fare.where(df.Pclass == 3).replace({np.nan: val})
        df["Fare"] = df["Fare"].fillna("13")

        cabin_grouping = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
        df['CabinGrouping'] = df['Cabin'].map(lambda x: group_by_sub_string(x, cabin_grouping))
        df.drop('Cabin', inplace=True, axis=1)
        df.drop('Ticket', inplace=True, axis=1)
        df['CabinGrouping'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'T': 6, 'G': 7, 'Unknown': 8}, inplace=True)
        df.drop('Age', inplace=True, axis=1)  # Look at fixing Nan Values here
        if train_set:
            if get_cv:
                train, cv = sklearn.model_selection.train_test_split(df, test_size=0.22)
                labels_train = train[['Survived']]
                labels_cv = cv[['Survived']]

                return [train, labels_train, cv, labels_cv]
            labels = df[['Survived']]
            df.drop('Survived', inplace=True, axis=1)
            return[df, labels]

        else:
            return df
