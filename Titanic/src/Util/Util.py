import json
import numpy as np
import pandas as pd

# import tensorflow as tf
import sklearn.model_selection
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import validation_curve, GridSearchCV
import bisect
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import seaborn as sns


def group_by_sub_string(original_string, list_of_sub_strings):
    for substring in list_of_sub_strings:
        if original_string.find(substring) != -1:
            return substring
    return np.nan


def extract_datasets(full_df):
    x_train = full_df.loc[full_df['Survived'].notnull()]
    x_test = full_df.loc[full_df['Survived'].isnull()]
    return x_train, x_test


def onehot_encoding(df, feature_list):
    try:
        df = pd.get_dummies(df, columns=feature_list)
        return df
    except:
        pass


# TODO Add plotting ROC curves


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
    cv_score = cross_val_score(grid_f, X, y)
    return [grid_f.best_params_, cv_score, grid_f.score(X, y)]


class Util:
    def __init__(self):
        self.survivalPerTicket = {}
        self.configValues = {}
        self.scalar = np.nan
        try:
            config_file = open("../config.JSON")
            self.configValues = json.load(config_file)
        except (OSError, FileNotFoundError) as e:
            self.configValues["env"] = "Kaggle"
        self.dataPath = "../../data" if self.configValues["env"] == "Local" else "/kaggle/input"

        self.fullDf = pd.concat([pd.read_csv(self.dataPath + '/train.csv'),
                                 pd.read_csv(self.dataPath + '/test.csv')], ignore_index=True)

        self.fullDf['Title'] = self.fullDf['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

        self.fullDf['Married'] = self.fullDf['Title'].map(lambda x: x == 'Mr' or x == 'Mrs')
        self.fullDf[['Title']] = self.fullDf[['Title']].replace(
            dict.fromkeys(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                          'Miss'))

        self.fullDf[['Title']] = self.fullDf[['Title']].replace(
            dict.fromkeys(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev']
                          , 'Special'))

        self.fullDf["Cabin"] = self.fullDf["Cabin"].fillna("Unknown")
        cabin_grouping = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
        self.fullDf['CabinGrouping'] = self.fullDf['Cabin'].map(lambda x: group_by_sub_string(x, cabin_grouping))
        self.fullDf['HasCabin'] = self.fullDf['CabinGrouping'].map(lambda x: 1 if x != 'Unknown' else 0)
        features_for_onehot_encoding = ["Title", "Pclass", "Embarked", "CabinGrouping"]

        # self.fullDf['Fare'] = pd.qcut(self.fullDf['Fare'], 9, labels=False, precision=0)  # Might not be needed for MLP

        self.fullDf['Sex'].replace({"female": 1, "male": 0}, inplace=True)

        # Dropping columns
        self.fullDf.drop('Name', inplace=True, axis=1)
        self.fullDf.drop('PassengerId', inplace=True, axis=1)
        self.fullDf.drop('Cabin', inplace=True, axis=1)
        # self.fullDf.drop('Age', inplace=True, axis=1)
        self.fullDf.drop('Ticket', inplace=True, axis=1)

        # filling null values
        self.fullDf["Embarked"] = self.fullDf["Embarked"].fillna("S")
        med_fare = self.fullDf.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
        self.fullDf['Fare'] = self.fullDf['Fare'].fillna(med_fare)
        self.fullDf = onehot_encoding(self.fullDf, features_for_onehot_encoding)

        self.fullDf['Fare'] = pd.qcut(self.fullDf['Fare'], 9, labels=False, precision=0)  # Might not be needed for MLP

        [x_train, x_test] = extract_datasets(self.fullDf)

        X_train_age = self.fullDf[[x for x in list(x_train) if not x in ["Survived"]]]  # select use features

        # split data for train
        X_predict_age = X_train_age.loc[X_train_age["Age"].isnull()]
        X_train_age = X_train_age.loc[X_train_age["Age"].notnull()]  # use rows which age is not null
        y_train_age = X_train_age.Age
        try:
            X_train_age.drop("Age", axis=1, inplace=True)
            X_predict_age.drop("Age", axis=1, inplace=True)
        except:
            print("except")

        age_Scalar = preprocessing.StandardScaler().fit(X_train_age)
        X_train_age = age_Scalar.transform(X_train_age)
        X_predict_age = age_Scalar.transform(X_predict_age)
        Age_None_list = self.fullDf[self.fullDf['Age'].isnull()].index.tolist()

        mlr = MLPRegressor(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(50, 50), random_state=1)
        mlr.fit(X_train_age, y_train_age)

        age_predictions = mlr.predict(X_predict_age).tolist()
        self.fullDf["Age"][Age_None_list] = age_predictions

        [self.x_train, self.x_test] = extract_datasets(self.fullDf)
        self.y_train = x_train[['Survived']]
        self.fullDf.drop('Survived', inplace=True, axis=1)


    def scale_df(self, X):
        if type(self.scalar) != preprocessing.StandardScaler:
            self.scalar = preprocessing.StandardScaler().fit(X)
            print('Creating scalar for training set')
        return pd.DataFrame(self.scalar.transform(X))

    def normalise(self, dataframe):
        pass

    def get_df(self):

        return [self.x_train, self.y_train, self.x_test]
