import json
import numpy as np
import pandas as pd


# import tensorflow as tf
import sklearn.model_selection


def group_by_sub_string(original_string, list_of_sub_strings):
    for substring in list_of_sub_strings:
        if original_string.find(substring) != -1:
            return substring
    return np.nan


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

    def get_df(self, dataset_name, train_set=True):
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
        df['CabinGrouping'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, \
                                     'E': 4, 'F': 5, 'T': 6, 'G': 7, 'Unknown': 8}, inplace=True)
        df.drop('Age', inplace=True, axis=1) # Look at fixing Nan Values here
        if train_set:
            train, cv = sklearn.model_selection.train_test_split(df, test_size=0.22)
            labels_train = train[['Survived']]
            labels_cv = cv[['Survived']]
            df.drop('Survived', inplace=True, axis=1)
            return [train, labels_train, cv, labels_cv]
        else:
            return df
