import numpy
import pandas
from seaborn.utils import sig_stars
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from TitanicMachineLearningfromDisaster import LogisticRegressionCS
from TitanicMachineLearningfromDisaster import PrepareDataTitanic
from TitanicMachineLearningfromDisaster import PrepareData
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


def featureAnalysisCorrelation(dataset):
    ordinal_features = ['Pclass', 'SibSp', 'Parch']
    nominal_features = ['Sex', 'Embarked']
    survived_feature = ['Survived']
    numerical_feature = ['Age', 'Fare']
    print('Correlation')
    correlation = dataset[ordinal_features + nominal_features + survived_feature + numerical_feature].corr().round(
        2);
    print(correlation)


def featureAnalysisPlots(dataset):


    sibSp=sns.factorplot(x='SibSp',y='Survived' ,data=dataset,kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();
    sibSp = sns.factorplot(x='Pclass', y='Survived', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();
    sibSp = sns.factorplot(x='Pclass', y='Survived',hue='Sex', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();
    sibSp = sns.factorplot(x='Parch', y='Survived', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();
    sibSp = sns.factorplot(x='Sex', y='Survived', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();
    sibSp = sns.factorplot(x='Embarked', y='Survived', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();
    sibSp = sns.factorplot('Pclass',col='Embarked',data=dataset, kind='count')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();
    age = sns.FacetGrid(train, col='Survived')
    age = age.map(sns.distplot, "Age")
    plt.show();
    g = sns.distplot(dataset["Fare"], color="m",label="Skewness : %.2f"%(dataset["Fare"].skew()))
    g = g.legend(loc="best")
    plt.show();

def featureAnalysisTable(dataset):
    print(dataset[['Sex','Survived']].groupby('Sex').mean());



def analyseMissingValues(dataset):
    print(dataset.isnull().sum())
    print(dataset.info())
    print(dataset.describe())
    print(dataset['Fare'].isnull().sum())
    print(dataset["Embarked"].isnull().sum())

def analyseAge(dataset):
    sibSp = sns.factorplot(x='Sex', y='Age', data=dataset, kind='box')
    sibSp = sibSp.set_ylabels("Age")
    plt.show();
    sibSp = sns.factorplot(x='SibSp', y='Age', data=dataset, kind='box')
    sibSp = sibSp.set_ylabels("Age")
    plt.show();
    sibSp = sns.factorplot(x='Parch', y='Age', data=dataset, kind='box')
    sibSp = sibSp.set_ylabels("Age")
    plt.show();
    sibSp = sns.factorplot(x='Pclass', y='Age', data=dataset, kind='box')
    sibSp = sibSp.set_ylabels("Age")
    plt.show();

def analyseTicket(dataset):
    sibSp = sns.factorplot(x='Ticket', y='Survived', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();