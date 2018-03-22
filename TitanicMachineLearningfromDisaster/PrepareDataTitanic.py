import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

def convertDataToNumericalDataFrame(train_data):
    #train_data = pandas.read_csv(dataset, index_col='PassengerId')
    #print(train_data)
    train_data['Sex']=numpy.where(train_data['Sex']=='male',1,0)
    train_data['Cabin'] = numpy.where(pandas.isnull(train_data['Cabin']), 1, 0)
    train_data['Embarked'] =numpy.where(train_data['Embarked'] == 'C', 1, (numpy.where(train_data['Embarked'] =='S', 0.5, 0)))
    return train_data;


def fillMissingData(dataset):
    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(numpy.nan)
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    return dataset



def logScaleFare(dataset):
    dataset['Fare']=dataset['Fare'].map(lambda x:numpy.log(x) if x>0 else 0)

def conditions(x):
    if x == 'S':
        return 1
    elif x == 'Q':
        return 0
    else:
        return -1





def convertDataToNumerical(dataset):
    dataset['Sex'] =dataset['Sex'].map({'male':0,'female':1})

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