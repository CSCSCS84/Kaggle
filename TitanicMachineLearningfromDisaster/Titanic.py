import numpy
import pandas
from seaborn.utils import sig_stars
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    return 1.0/(1+numpy.exp(-x))

def calcResult(probability,test_data):

    print("in calcresult")
    print(probability)
    print(testdata.shape)
    print(numpy.where(probability['Survived'] < 0.5, 0, 1))
    test_data['Survived'] = numpy.where(probability['Survived'] < 0.5, 0, 1)
    return test_data;

def conditions(x):
    if x == 'S':
        return 1
    elif x == 'Q':
        return 0
    else:
        return -1

def convertDataToNumerical():
    train_data = pandas.read_csv('train.csv', index_col='PassengerId')
    n=train_data.shape[1];
    m=train_data.shape[0];
    skippedColumns=0;
    unusedFeatures=2;
    XX=pandas.DataFrame();
    XX = numpy.zeros(([m, n-unusedFeatures]), dtype=float)
    XX[:,0]=train_data.iloc[0:,0+skippedColumns]
    XX[:, 1] = train_data.iloc[0:, 1+skippedColumns];
    skippedColumns=skippedColumns+1;
    XX[:, 2] = numpy.where(train_data['Sex']=='male',1,-1)
    XX[:, 3] = train_data.iloc[0:, 3+skippedColumns];
    XX[:, 4] = train_data.iloc[0:, 4 + skippedColumns];
    XX[:, 5] = train_data.iloc[0:, 5 + skippedColumns];
    skippedColumns = skippedColumns + 1;
    XX[:, 6] = train_data.iloc[0:, 6 + skippedColumns];
    #TODO Cabin is not yet transformed correctly
    XX[:, 7] = numpy.where(train_data['Cabin']=='',1,-1)
    XX[:, 8]=numpy.where(train_data['Embarked'] == 'C', 1,
         (numpy.where(train_data['Embarked'] =='S', 0, -1)))

    return XX;

def convertDataToNumericalDataFrame():
    train_data = pandas.read_csv('train.csv', index_col='PassengerId')
    train_data['Sex']=numpy.where(train_data['Sex']=='male',1,0)
    train_data['Cabin'] = numpy.where(pandas.isnull(train_data['Cabin']), 1, 0)
    train_data['Embarked'] =numpy.where(train_data['Embarked'] == 'C', 1, (numpy.where(train_data['Embarked'] =='S', 0.5, 0)))
    return train_data;

def scaleData(train_data):
    scaler = MinMaxScaler()
    featuresToScale=['Pclass','Age','SibSp','Parch','Fare']
    train_data[featuresToScale] = scaler.fit_transform(train_data[featuresToScale])
    return train_data;


def prepareData(features,train_data):
    numberOfFeatures = len(features)

    XX[:, 0] = train_data[0:, 1];
    XX[:, 1] = [1 if 'male' else -1 for x in train_data[0:, 3]];
    XX[:, 2] = train_data[0:, 4];
    XX[:, 3] = train_data[0:, 5];
    XX[:, 4] = train_data[0:, 6];
    XX[:, 5] = train_data[0:, 8];
    where_are_NaNs = numpy.isnan(XX)
    XX[where_are_NaNs] = 0
    return XX;



def calcThetaLogisticRegression(train_data,features):
    alpha = 0.001;
    numOfSteps=2000;

    numberOfRemarks=len(features);
    m=numberOfRemarks;
    theta=pandas.DataFrame(numpy.ones((m, 1)),index=features)
    n=train_data.shape[1];
    #y=numpy.zeros(([n,1]),dtype=float)

    y=pandas.DataFrame(train_data['Survived']);

    XX=train_data[features]
    #print(XX)
    XXtranpose=XX.transpose();
    #print("Shapes")
    for i in range(1,numOfSteps):
        left = XXtranpose.multiply(alpha/m);
        sigMat = XX.dot(theta);
        right = sigmoid(sigMat).values-(y);
        theta = theta.values-(left.dot(right));
    return theta;

#use theta to prognose
def calcPredictionDataframe(theta,features):
    testdata = pandas.read_csv('test.csv', index_col='PassengerId')
    #testdata = testdata.dropna()

    testdata['Sex'] = numpy.where(testdata['Sex'] == 'male', 1, 0)
    testdata['Embarked'] = numpy.where(testdata['Embarked'] == 'C', 1,
                                         (numpy.where(testdata['Embarked'] == 'S', 0.5, 0)))
    XX = testdata[features]
    print("Calcprediction")
    print(XX.shape)
    prob=sigmoid(XX.dot(theta))
    return prob;


def calcPrediction(theta):
    testdata = pandas.read_csv('test.csv',index_col='PassengerId')
    numberOfRemarks=7;

    testdata=numpy.array(testdata)

    n=testdata.shape[0]
    XT=numpy.zeros(([n,numberOfRemarks]),dtype=float)
    XT[:,0]=testdata[0:,0];
    XT[:,1]=[1 if 'male' else -1 for x in testdata[0:,2]];
    XT[:,2]=testdata[0:,3];
    XT[:,3]=testdata[0:,4];
    XT[:,4]=testdata[0:,5];
    XT[:,5]=testdata[0:,7];

    where_are_NaNs = numpy.isnan(XT)
    XT[where_are_NaNs] = 0

    probability=sigmoid(numpy.dot(XT,theta))
    return probability;

def calcSurvived():
    theta=calcThetaLogisticRegression();
    probability=calcPrediction(theta);
    result=calcResult(probability)
    return result;




train_data=convertDataToNumericalDataFrame()
train_data=train_data.dropna()

train_data=scaleData(train_data)

ordinal_features = ['Pclass', 'SibSp', 'Parch']
nominal_features = ['Sex', 'Embarked']
survived_feature=['Survived']
numerical_feature=['Age','Fare']
correlation=train_data[ordinal_features + nominal_features+survived_feature+numerical_feature].corr().round(2);
print(correlation)

features=['Sex', 'Embarked']
theta=calcThetaLogisticRegression(train_data,features);
print("Theta")
print(theta)
probability=calcPredictionDataframe(theta,features)
print(probability)
testdata = pandas.read_csv('test.csv',index_col='PassengerId')
result=calcResult(probability,testdata)
print(result)
#numberSurvived=sum(1 if i < 0.5 else 0 for i in probability)
#print(numberSurvived)
#print(theta)

#print(probability)

resultSurvived = pandas.DataFrame(result['Survived'])
#resultSurvived['PassengerId']=testdata['PassengerId']

print(resultSurvived)
#resultSurvived['Survived']=testdata['Survived']

#print(resultSurvived)
#numpy.savetxt(r'Calcedresults.csv', resultSurvived.values, fmt='%s',header='PassengerId\tSurvived',delimiter='\t',comments="")
#numpy.savetxt(r'Realresults.csv', result.values, fmt='%s',header='PassengerId\tSurvived\tName',delimiter='\t',comments="")

resultSurvived.to_csv('Calcedresults.csv', header='PassengerId\tSurvived',sep=',')