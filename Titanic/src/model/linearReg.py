# import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from Titanic.src.Util.Util import Util


def main():
    util = Util()
    [X, y, x_cv, y_cv] = util.get_df('/train.csv')
    y = np.ravel(y)
    X_test = util.get_df('/test.csv', False)
    df = pd.read_csv('../../data/test.csv')
    Ids = df[["PassengerId"]]
    # util.convert_to__normalised_tensor(input_data)
    # print(X.isnull().sum())
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    # print(X_test.isnull().sum())
    # print(X_test[X_test.isna().any(axis=1)])
    predictions = model.predict(X_test)

    #
    output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': predictions})
    print(output)
    output.to_csv('my_submissionYes.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == "__main__":
    main()
