import json
import string


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def group_by_sub_string(original_string, list_of_sub_strings):
    for substring in list_of_sub_strings:
        if original_string.find(substring) != -1:
            return substring
    return np.nan


configValues = {}
try:
    configFile = open("../config.JSON")
    configValues = json.load(configFile)
except (OSError, FileNotFoundError) as e:
    configValues["env"] = "Kaggle"
dataPath = "../../data" if configValues["env"] == "Local" else "/kaggle/input"

df = pd.read_csv(dataPath + "/train.csv")

# Look at age values and name to check for missing values
df[['Name', 'Age']].to_csv('Age Values.txt')

df['Fare'] = pd.qcut(df['Fare'], 13, labels=False, precision=0)

# Determining number of bins to use for Age  correlation of 0.049151293565911795
currentCorr = 0
optimalAgeBin = 0
for i in range(10):
    try:
        testDf = df['Age'].copy()
        df['Age'] = pd.qcut(df['Age'], i, labels=False, precision=0)
        ageCorr = df.corr()['Survived'].sort_values(ascending=False)['Age']
        df['Age'] = testDf
        if (ageCorr > currentCorr):
            currentCorr = ageCorr
            optimalAgeBin = i
    except:
        print('Exception found with binning')
print("optimal age bin %s, correlation %s" % (str(optimalAgeBin), str(currentCorr)))


#  Determining optimal Fare bin
currentCorr = 0
optimalAgeBin = 0
for i in range(10):
    try:
        testDf = df['Fare'].copy()
        df['Fare'] = pd.qcut(df['Fare'], i, labels=False, precision=0)
        ageCorr = df.corr()['Survived'].sort_values(ascending=False)['Fare']
        df['Fare'] = testDf
        if (ageCorr > currentCorr):
            currentCorr = ageCorr
            optimalAgeBin = i
    except:
        print('Exception found with binning')
print("optimal Fare bin %s, correlation %s" % (str(optimalAgeBin), str(currentCorr)))

# Determining if thresholding Age is better, correlation 0.049151293565911795
currentCorr = 0
optimalAgeBin = 0
for i in range(50):
    try:
        testDf = df['Age'].copy()
        df['Age'] = df['Age'].map(lambda x: x > i)
        ageCorr = df.corr()['Survived'].sort_values(ascending=False)['Age']
        df['Age'] = testDf
        if (ageCorr > currentCorr):
            currentCorr = ageCorr
            optimalAgeBin = i
    except:
        print('Exception found with binning')
print("optimal age thresh for binary binning %s, correlation %s" % (str(optimalAgeBin), str(currentCorr)))

print(df['Age'].value_counts())

df['Age'] = pd.qcut(df['Age'], 10, labels=False, precision=0)

f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()
#  Plot to view Age Bins
# fig, axs = plt.subplots()
# sns.countplot(x='Age', hue='Survived', data=df)
#
# plt.xlabel('Age', size=15, labelpad=20)
# plt.ylabel('Passenger Count', size=15, labelpad=20)
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)
#
# plt.legend(['Not Survived', 'Survived'])
# plt.title('Survival Counts in Age Parameter')
# plt.show()

# View correlation matrix for data
# f, ax = plt.subplots(figsize=(10, 8))
# corr = df.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)
# plt.title("Correlation matrix for features in dataset: " + dataset_name)
# plt.show()


#  Plot to view Marriage Bins
fig, axs = plt.subplots()
sns.countplot(x='Married', hue='Survived', data=df)

plt.xlabel('Title', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'])
plt.title('Survival Counts in Married Parameter')
plt.show()

#  Plot to view Title Bins
fig, axs = plt.subplots()
sns.countplot(x='Title', hue='Survived', data=df)

plt.xlabel('Title', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'])
plt.title('Survival Counts in Title Parameter')
plt.show()

df['Surname'] = df['Name'].map(lambda x: x.split(',')[0])
df[['Surname']].to_csv('Surname.txt')

df.drop('Name', inplace=True, axis=1)
genAnalysisFile = 'generalAnalysis.txt'

# View prices for passenger ID's
TicketPrice, ax = plt.subplots()
ax = df.plot.scatter(y='Fare', x='PassengerId')
plt.show()

# View prices for Class of ticket
TicketPrice, ax = plt.subplots()
ax = df.plot.scatter(y='Fare', x='Pclass')
plt.show()

df['Title'] = df['Name'].map(lambda x: x.split(',')[1].split('.')[0])
print(df[['Title']])
print(df['Title'].value_counts())


df.drop('PassengerId', inplace=True, axis=1)
df['Sex'].replace({"female": 1, "male": 0}, inplace=True)
df['Embarked'].replace({"S": 3, "C": 1, "Q": 0}, inplace=True)


labels = df[['Survived']]
df.drop('Survived', inplace=True, axis=1)

dfTicketCount = df['Ticket'].value_counts()




# Grouping By Cabin and removing Nan Values from Cabin column
df["Cabin"] = df["Cabin"].fillna("Unknown")
CabinGrouping = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
df.groupby(['Cabin', 'Pclass']).size().to_csv('CheckCabins.txt', header=True)
df['CabinGrouping'] = df['Cabin'].map(lambda x: group_by_sub_string(x, CabinGrouping))
df.groupby(['CabinGrouping', 'Pclass']).size().to_csv('CabinGrouping.txt', header=True)
print(df.groupby(['Cabin', 'Pclass']).size(), file=open('CheckCabins', 'a'))
print('Number of unique ticket numbers: ' + str(df['Ticket'].value_counts().gt(1).sum()), \
      file=open(genAnalysisFile, 'a'))

SibFig, ax = plt.subplots()
siblingHist = df.hist('SibSp', ax=ax)
SibFig.savefig("Siblings.png", dpi=100)


ParFig, ax = plt.subplots()
siblingHist = df.hist('Parch', ax=ax)
ParFig.savefig("Parents.png", dpi=100)


# Uncomment when you want to run this analysis

print('Number of survivors', file=open(genAnalysisFile, 'a'))
labels.sum().to_csv(genAnalysisFile, mode='a', header=False)
print('Number of passengers in dataset: ' + str(labels.shape[0]), file=open(genAnalysisFile, 'a'))


# Uncomment when you want to run this analysis
df.isna().sum().to_csv(r'ColumnNanAnalysis', header=True)
