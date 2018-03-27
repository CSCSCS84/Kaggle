import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def generalAnalysis(dataset):
    print(dataset.isnull().sum())
    print(dataset.info())
    print(dataset.describe())


def printCorrelation(dataset):
    ordinal_features = ['Pclass', 'SibSp', 'Parch']
    nominal_features = ['Sex', 'Embarked']
    survived_feature = ['Survived']
    numerical_feature = ['Age', 'Fare']
    correlation = dataset[ordinal_features + nominal_features + survived_feature + numerical_feature].corr().round(
        2);
    print(correlation)


def plotSurvivalProbability(dataset, features):
    for f in features:
        plot = sns.factorplot(x=f, y='Survived', data=dataset, kind='bar')
        plot = plot.set_ylabels("survival probability")


def plotSurvivalProbabilityHue(dataset, features, hue):
    for f in features:
        plot = sns.factorplot(x=f, y='Survived', hue=hue, data=dataset, kind='bar')
        plot = plot.set_ylabels("survival probability")
        plt.show()


def plotSkewness(dataset, features):
    for f in features:
        plot = sns.distplot(dataset[f], color="m", label="Skewness : %.2f" % (dataset[f].skew()))
        plot = plot.legend(loc="best")
        plt.show()


def analyseAge(dataset, features):
    for f in features:
        plot = sns.factorplot(x=f, y='Age', data=dataset, kind='box')
        plot = plot.set_ylabels("Age")
        plt.show()


def survivalProbabilityGroupBySex(dataset):
    print(dataset[['Sex', 'Survived']].groupby('Sex').mean());


train = pandas.read_csv("Data/Input/train.csv", index_col='PassengerId')
test = pandas.read_csv("Data/Input/test.csv", index_col='PassengerId')
features = ['SibSp', 'Pclass', 'Parch', 'Sex', 'Embarked']

generalAnalysis(train)
printCorrelation(train)
survivalProbabilityGroupBySex(train)
analyseAge(train, features=['Sex', 'SibSp', 'Parch', 'Pclass'])
plotSurvivalProbability(train, features)
plotSurvivalProbabilityHue(train, features, hue='Sex')
plotSkewness(train, ['Fare'])
analyseAge(train, features=['Sex', 'SibSp', 'Parch', 'Pclass'])
