import numpy
import pandas
from seaborn.utils import sig_stars
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from TitanicMachineLearningfromDisaster import LogisticRegressionMy
from TitanicMachineLearningfromDisaster import PrepareData
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

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
    return dataset

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

    #print(dataset['Age'].isnull().sum())
    missingAge=dataset[dataset['Age'].isnull()];
    #print(missingAge)
    mean=dataset['Age'].median()
    for index,row in missingAge.iterrows():
        pclass=row['Pclass']
        sibsp=row['SibSp']
        parch=row['Parch']
        pclassMean=dataset[(dataset['Pclass']==pclass) & (dataset['SibSp']==sibsp) & (dataset['Parch']==parch)]['Age'].median();

        if numpy.isnan(pclassMean):
            dataset['Age'].ix[index]=mean;
        else:
            dataset['Age'].ix[index]=pclassMean

#analyse names that contain master or
def analyseNameAndAddTitle(dataset):
    names=dataset['Name']
    secondNames=[i.split(',')[1] for i in dataset['Name']]
    titles=[i.split('.')[0].strip() for i in secondNames ]
    dataset['Title']=titles;
    #print(dataset['Title'].unique())
    dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2,
                                             'Lady':3, 'the Countess':3, 'Countess':3, 'Capt':3, 'Col':3, 'Don':3, 'Dr':3, 'Major':3,
                                              'Rev':3, 'Sir':3, 'Jonkheer':3, 'Dona':3})
    dataset["Title"] = dataset["Title"].astype(int)


def analyseFamilySize(dataset):
    familysize=dataset['SibSp']+dataset['Parch']+1;

    dataset['Single']=familysize.map(lambda  s: 1 if s==1 else 0)
    dataset['SmallFamily'] = familysize.map(lambda s: 1 if s ==2 else 0)
    dataset['MediumFamily'] = familysize.map(lambda s: 1 if  2 < s <= 4 else 0)
    dataset['LargeFamily'] =familysize.map(lambda s: 1 if s >= 5 else 0)
    #print(dataset)

def convertIndicatorValues(dataset):
    dataset=pandas.get_dummies(dataset,columns=['Embarked'],prefix='EM')
    dataset = pandas.get_dummies(dataset, columns=['Title'])
    dataset = pandas.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
    return dataset;

def analysefamilySize(dataset):
    #print(dataset['Title'].unique())
    sibSp = sns.countplot(x='Title', data=dataset)
    plt.show()
    sibSp = sns.factorplot(x='Title', y='Survived', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();

def prepareCabin(dataset):
    dataset['Cabin']= dataset['Cabin'].fillna('X')
    dataset['Cabin']=dataset['Cabin']
    dataset['Cabin']=dataset['Cabin'].map(lambda s: s[0])
    dataset = pandas.get_dummies(dataset, columns=['Cabin'], prefix='C')

def prepareTicket(dataset):
    #print(dataset['Ticket'].unique())
    tickets=[]
    for f in dataset['Ticket']:
        #print(f)
        t=f.split(" ")[0]
        t=t.replace("/","")
        t = t.replace(".", "")

        if t.isdigit():
            t='X'
        #else:
        #    t=t[0]
        #if t[0]=='S':
        #    tickets.append(t)
        #else:
        tickets.append(t)
    dataset['Ticket']=tickets;
    dataset = pandas.get_dummies(dataset, columns=['Ticket'], prefix='T')
    return dataset;

def analyseTicket(dataset):
    sibSp = sns.factorplot(x='Ticket', y='Survived', data=dataset, kind='bar')
    sibSp = sibSp.set_ylabels("survival probability")
    plt.show();


def analyseData(dataset):
    analyseMissingValues(dataset)
    featureAnalysisPlots(train)
    # print(dataset)
    featureAnalysisPlots(train)
    featureAnalysisTable(dataset)
    analyseAge(dataset)


def prepareData(dataset):
    dataset = fillMissingData(dataset)
    fillMissingAge(dataset)
    convertDataToNumerical(dataset)
    logScaleFare(dataset)
    PrepareData.scaleData(dataset,['Fare','Age'])
    analyseNameAndAddTitle(dataset)
    analyseFamilySize(dataset)
    prepareCabin(dataset)
    dataset = prepareTicket(dataset)
    dataset = convertIndicatorValues(dataset)
    return dataset

features=["Age","SibSp","Parch","Fare"]

train = pandas.read_csv("Input/train.csv",index_col='PassengerId')
test = pandas.read_csv("Input/test.csv",index_col='PassengerId')
train[["Age","SibSp","Parch","Fare"]]=train[["Age","SibSp","Parch","Fare"]].astype(float)
test[["Age","SibSp","Parch","Fare"]]=test[["Age","SibSp","Parch","Fare"]].astype(float)
outliers=detectOutliners(train,2,features)
train=train.drop(outliers)

dataset=pandas.concat([train,test],axis=0)
#print(dataset)



dataset=prepareData(dataset)
#dataset=prepareTicket(dataset)


#print(dataset)
trainDataSet=dataset[dataset['Survived'].notna()]

testdata=dataset[dataset['Survived'].isna()]
#print(trainDataSet)
#features=["Age",'Fare','Pclass','Sex','Single', 'SmallFamily', 'MediumFamily', 'LargeFamily','EM_C', 'EM_Q', 'EM_S','Title_0', 'Title_1', 'Title_2', 'Title_3']
#features=['Age','Fare','Pclass','Sex']
#print(list(dataset.columns.values))

features=['Age','Fare',  'Sex', 'Single', 'SmallFamily', 'MediumFamily', 'LargeFamily', 'EM_C', 'EM_Q', 'EM_S', 'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Pc_1', 'Pc_2', 'Pc_3']
correlation = dataset[features].corr().round(
    2);
#print(correlation)
classifierCS=LogisticRegressionMy.LogisticRegressionCS(max_iter=100);
result=classifierCS.logRegression(trainDataSet,testdata,features)
resultSurvived = pandas.DataFrame(result['Survived'])
resultSurvived.to_csv('Calcedresults.csv', header='PassengerId\tSurvived',sep=',')


X_train=trainDataSet[features]
y_train=trainDataSet['Survived']
classifier = LogisticRegression(random_state=0, max_iter=10000, fit_intercept=False,tol=0.00001)
classifier.fit(X_train, y_train)

X_test=testdata[features]

print(classifier)
#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
#          verbose=0, warm_start=False)


y_predTest = classifier.predict(X_test)
y_predTrain=classifier.predict(X_train)
confusion_matrix = confusion_matrix(y_train, y_predTrain)

print(classifier.coef_)
#print(confusion_matrix)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(classifier.score(X_train, y_train)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_predTest)))

error=classifierCS.calcError(features,y_train,classifier.coef_.transpose(),X_train)
print(error)
#print(y_pred)
#resultSurvived = pandas.DataFrame(y_pred)
#resultSurvived.to_csv('Calcedresults.csv', header='PassengerId\tSurvived',sep=',')