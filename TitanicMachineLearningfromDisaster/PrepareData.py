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


def scaleData(train_data,featuresToScale):
    #print(train_data)
    scaler = MinMaxScaler()

    train_data[featuresToScale] = scaler.fit_transform(train_data[featuresToScale])
    return train_data;

def conditions(x):
    if x == 'S':
        return 1
    elif x == 'Q':
        return 0
    else:
        return -1

#def prepareTestdata(testdata):
#    testdata['Sex'] = numpy.where(testdata['Sex'] == 'male', 1, 0)
#    testdata['Embarked'] = numpy.where(testdata['Embarked'] == 'C', 1,
#                                           (numpy.where(testdata['Embarked'] == 'S', 0.5, 0)))