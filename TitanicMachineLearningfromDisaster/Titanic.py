import numpy
import pandas
from TitanicMachineLearningfromDisaster import TitanicYassineGhouzam
from TitanicMachineLearningfromDisaster import PrepareData
from TitanicMachineLearningfromDisaster import LogisticRegression
from seaborn.utils import sig_stars

trainDataSet = pandas.read_csv("Input/train.csv")
features=["Age","SibSp","Parch","Fare"]
featuresToScale = ['Pclass', 'Age', 'SibSp', 'Parch','Fare']
outliners=TitanicYassineGhouzam.detectOutliners(trainDataSet,2,features)
trainDataSet=trainDataSet.drop(outliners)

trainDataSet=PrepareData.convertDataToNumericalDataFrame(trainDataSet)

TitanicYassineGhouzam.fillMissingData(trainDataSet)
trainDataSet=trainDataSet.dropna()
TitanicYassineGhouzam.logScaleFare(trainDataSet)
trainDataSet=PrepareData.scaleData(trainDataSet,featuresToScale)


testdataSet='Input/test.csv';
testdata = pandas.read_csv(testdataSet,index_col='PassengerId')

testdata=PrepareData.convertDataToNumericalDataFrame(testdata)
testdata=testdata.fillna(testdata.mean())

TitanicYassineGhouzam.logScaleFare(testdata)
testdata=PrepareData.scaleData(testdata,featuresToScale)

print(testdata)
result=LogisticRegression.logRegression(trainDataSet,testdata,features)
resultSurvived = pandas.DataFrame(result['Survived'])
resultSurvived.to_csv('Calcedresults.csv', header='PassengerId\tSurvived',sep=',')