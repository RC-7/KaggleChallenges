import json
import numpy as np
import pandas as pd

# import tensorflow as tf
import sklearn.model_selection
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import validation_curve, GridSearchCV

import seaborn as sns


def group_by_sub_string(original_string, list_of_sub_strings):
    for substring in list_of_sub_strings:
        if original_string.find(substring) != -1:
            return substring
    return np.nan


# TODO Add plotting ROC curves

def scale_df(X):
    scalar = preprocessing.StandardScaler().fit(X)
    return pd.DataFrame(scalar.transform(X))


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

    plt.title("Validation Curve with k means" + "(" + param_name + ")")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
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
        self.survivalPerTicket = {}
        self.configValues = {}
        try:
            config_file = open("../config.JSON")
            self.configValues = json.load(config_file)
        except (OSError, FileNotFoundError) as e:
            self.configValues["env"] = "Kaggle"
        self.dataPath = "../../data" if self.configValues["env"] == "Local" else "/kaggle/input"

        self.fullDf = pd.concat([pd.read_csv(self.dataPath + '/train.csv'),
                                 pd.read_csv(self.dataPath + '/test.csv')])

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
        self.fullDf['Surname'] = self.fullDf['Name'].map(lambda x: x.split(',')[0])
        df['Surname'] = df['Name'].map(lambda x: x.split(',')[0])
        surname_count = self.fullDf.groupby('Surname').size()
        ticket_count = self.fullDf.groupby('Ticket').size()
        # print(ticket_count)
        # print(ticket_count)


        df['Title'] = df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
        df['Married'] = df['Title'].map(lambda x: x == 'Mr' or x == 'Mrs')
        df[['Title']] = df[['Title']].replace(
            dict.fromkeys(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                          'Miss'))
        df[['Title']] = df[['Title']].replace(
            dict.fromkeys(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev']
                          , 'Special'))

        df['Title'].replace({'Mr': 0, 'Miss': 1, 'Master': 2, 'Special': 3},
                                    inplace=True)

        df['SurnameCount'] = df['Surname'].map(lambda x: surname_count[x])

        df['Ticket_Frequency'] = df['Ticket'].map(lambda x: ticket_count[x])

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

        cabin_grouping = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
        df['CabinGrouping'] = df['Cabin'].map(lambda x: group_by_sub_string(x, cabin_grouping))
        df['HasCabin'] = df['CabinGrouping'].map(lambda x: 1 if x != 'Unknown' else 0)
        df.drop('Cabin', inplace=True, axis=1)

        df['CabinGrouping'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'T': 0, 'G': 7, 'Unknown': 8},
                                    inplace=True)
        # Fill Age Nan's
        df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

        med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
        # Filling the missing value for one missing passenger
        df['Fare'] = df['Fare'].fillna(med_fare)

        df['Fare'] = pd.qcut(df['Fare'], 9, labels=False, precision=0)
        df['Age'] = pd.qcut(df['Age'], 2, labels=False, precision=0)
        if train_set:
            self.survivalPerTicket = df.groupby('Ticket')['Survived'].mean()
            df['Ticket_percentage_survival'] = df['Ticket'].map(lambda x: self.survivalPerTicket[x] / ticket_count[x])
            df['Know_Ticket_Survival_percentage'] = df['Ticket'].map(lambda x: 0 if x in self.survivalPerTicket else 1)
            df.drop('Ticket', inplace=True, axis=1)
            if get_cv:
                train, cv = sklearn.model_selection.train_test_split(df, test_size=0.22)
                labels_train = train[['Survived']]
                labels_cv = cv[['Survived']]

                return [train, labels_train, cv, labels_cv]
            labels = df[['Survived']]
            df.drop('Survived', inplace=True, axis=1)
            return [df, labels]

        else:
            # Use the mean survival percentage if the ticket was not in the training set.
            #  This is making model predict everyone lives
            df['Ticket_percentage_survival'] = df['Ticket'].map(lambda x: self.survivalPerTicket[x] / ticket_count[x] \
                if x in self.survivalPerTicket else self.survivalPerTicket.mean())
            df['Know_Ticket_Survival_percentage'] = df['Ticket'].map(
                lambda x: 0 if x in self.survivalPerTicket else 1)
            df.drop('Ticket', inplace=True, axis=1)
            return df
