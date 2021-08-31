___Titanic submissions___
===========================
__Scores for the [Titanic Kaggle competition](https://www.kaggle.com/c/titanic/overview)__
-
| Model | Best Score (On Test set)    | Parameters | Features |
| ----- | ----- | ---------- | -------- |
| Neural Network | 80.861 % | {'activation': 'tanh', 'alpha': 0.02, 'hidden_layer_sizes': (17,), 'learning_rate': 'constant', 'max_iter': 20, 'solver': 'lbfgs', 'random_state': 1} | ['Sex', 'Age', 'Fare', 'Married', 'HasCabin', 'Title_Master','Title_Miss', 'Title_Mr', 'Title_Special', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S'] |
| Random Forrest | 78.708 % | {'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100,'random_state': 100, 'max_features': 'auto', 'criterion': 'entropy'} | ["Sex", "Title", "Fare", "HasCabin", "Pclass", "Married", "Embarked", "Ticket_Frequency", "SibSp", "CabinGrouping", "Parch" ] |
| Kmeans | 77.511 % | {'n_clusters': 20, 'n_init': 100, 'max_iter': 1000, 'init': 'random'}|["Sex", "Title", "Fare", "HasCabin", "Pclass", "Married", "Embarked", "Ticket_Frequency", "SibSp", "CabinGrouping", "Parch" ] |

## __Brief Summary of feature Engineering employed in the best submissions__
- One hot encoding for a number of the features that could be grouped
- Binned Fare and Age features based on the best correlated number of bins. The bins used were found by performing K means clustering on the features.
- Calculate missing Age data based on a regression classifier trained on the rest of the data.
- Filled missing fare data based on mean for class of passenger missing data.
- Filled missing Embarked data based on manually evaluating passenger data.
- Grouped Cabin blocks based on Cabin data given.
- Grouped passengers with similar Titles.

## __On Runing the code__
- To run a model change the value of `tuning` and `predict` in config.JSON to either look at tuning Hyperparameters or use the Hyperparameters hardcoded in the model file to: train a model, predict data based on the testing dataset and create a submission file. 


[comment]: <> (## __Brief Summary of recon done on data__)