import numpy
import pandas
from seaborn.utils import sig_stars
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

#detect outliners using Tuking method
def detectOutliners(train,numOfOutlines,features):
    outliners=[];
    firstQuantil=train.quantile(0.25)
    thirdQuantil=train.quantile(0.75)
    IQR=thirdQuantil-firstQuantil;
    lower=firstQuantil-1.5*IQR;
    upper=thirdQuantil+1.5*IQR;

    outliers=[];
    for f in features:
        firstQuantil = train[f].quantile(0.25,interpolation='linear')
        thirdQuantil = train[f].quantile(0.75,interpolation='linear')
        IQR = thirdQuantil - firstQuantil;
        lower = firstQuantil - 1.5 * IQR;
        upper = thirdQuantil + 1.5 * IQR;
        outliersfeature=train[(train[f]<lower) | (train[f]>upper)].index.values;
        outliers.extend(outliersfeature)
    outliersCount=Counter(outliers)

    multiple_outliers= list( k for k, v in outliersCount.items() if v > numOfOutlines )
    return multiple_outliers

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



def logScaleFare(dataset):
    dataset['Fare']=dataset['Fare'].map(lambda x:numpy.log(x) if x>0 else 0)

def fillMissingData(dataset):
    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(numpy.nan)
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

def analyseMissingValues(dataset):
    print(dataset.isnull().sum())
    print(train.info())
    print(train.describe())
    print(dataset['Fare'].isnull().sum())
    print(dataset["Embarked"].isnull().sum())

def convertDataToNumerical(dataset):
    dataset['Sex'] =dataset['Sex'].map({'male':0,'female':1})

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

def fillMissingAge(dataset):

    print(dataset['Age'].isnull().sum())
    missingAge=dataset[dataset['Age'].isnull()];
    print(missingAge)
    mean=dataset['Age'].mean()
    for index,row in missingAge.iterrows():
        pclass=row['Pclass']
        sibsp=row['SibSp']
        parch=row['Parch']
        print(sibsp)
        print(parch)
        #print(dataset[dataset.iloc[d]['Pclass']==dataset['Pclass']])
        pclassMean=dataset[dataset['Pclass']==pclass & dataset['SibSp']==sibsp & dataset['Parch']==parch]['Age'].median();
        #print(pclassMean)

features=["Age","SibSp","Parch","Fare"]

train = pandas.read_csv("Input/train.csv")
test = pandas.read_csv("Input/test.csv",index_col='PassengerId')
train[["Age","SibSp","Parch","Fare"]]=train[["Age","SibSp","Parch","Fare"]].astype(float)
test[["Age","SibSp","Parch","Fare"]]=test[["Age","SibSp","Parch","Fare"]].astype(float)
outliers=detectOutliners(train,2,features)
train=train.drop(outliers)

dataset=pandas.concat([train,test],axis=0)
#print(dataset)

analyseMissingValues(dataset)
fillMissingData(train)


#featureAnalysisPlots(train)

#log scaling of Fare because of high skewness
logScaleFare(dataset)
convertDataToNumerical(dataset)
#print(dataset)
#featureAnalysisPlots(train)
#featureAnalysisTable(dataset)


#analyseAge(dataset)

fillMissingAge(dataset)