import numpy
import pandas
from TitanicMachineLearningfromDisaster import TitanicYassineGhouzam
from TitanicMachineLearningfromDisaster import PrepareDataTitanic
from TitanicMachineLearningfromDisaster import LogisticRegressionCS
from seaborn.utils import sig_stars

#def prepareTitanicData():


trainDataSet = pandas.read_csv("Input/train.csv")
features=["Age","SibSp","Parch","Fare"]
featuresToScale = ['Pclass', 'Age', 'SibSp', 'Parch','Fare']
outliners=TitanicYassineGhouzam.detectOutliners(trainDataSet,2,features)
trainDataSet=trainDataSet.drop(outliners)
trainDataSet=trainDataSet.dropna()


trainDataSet=PrepareDataTitanic.convertDataToNumericalDataFrame(trainDataSet)

TitanicYassineGhouzam.fillMissingData(trainDataSet)

TitanicYassineGhouzam.logScaleFare(trainDataSet)
trainDataSet=PrepareDataTitanic.scaleData(trainDataSet, featuresToScale)


testdataSet='Input/test.csv';
testdata = pandas.read_csv(testdataSet,index_col='PassengerId')

testdata=PrepareDataTitanic.convertDataToNumericalDataFrame(testdata)
testdata=testdata.fillna(testdata.mean())

TitanicYassineGhouzam.logScaleFare(testdata)
testdata=PrepareDataTitanic.scaleData(testdata, featuresToScale)

print(testdata)
result=LogisticRegressionCS.logRegression(trainDataSet, testdata, features)
resultSurvived = pandas.DataFrame(result['Survived'])
resultSurvived.to_csv('Calcedresults.csv', header='PassengerId\tSurvived',sep=',')