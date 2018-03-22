import numpy
import pandas


wrongPrediction = pandas.read_csv("WrongPrediction.csv", index_col='PassengerId')
train = pandas.read_csv("PreparedTrain.csv", index_col='PassengerId')
wrongDecribe=wrongPrediction.describe()
trainDescribe=train.describe()
features=['Age','Fare',  'Sex', 'Single', 'SmallFamily', 'MediumFamily', 'LargeFamily', 'EM_C', 'EM_Q', 'EM_S', 'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Pc_1', 'Pc_2', 'Pc_3']

for f in features:
    print(wrongDecribe[f])
    print(trainDescribe[f])
    print("")
