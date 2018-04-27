from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
import pandas
import numpy

fileNameExtension = 'ABCEFGHIJKPQRTU12'
fileNameExtensionTest = 'ABCEFGHIJKPQRTU1'
featureNumber = 2
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)

test = titanic.test
result = pandas.DataFrame(index=test.index)
result['Survived'] = numpy.ones((test.shape[0], 1)) * (-1)

allReadyPredicted = set([])

for index, row in test.iterrows():
    pclass1 = row['Pclass_1']
    pclass3 = row['Pclass_3']
    sex = row['Sex']
    LargeGroup = row['LargeGroup']
    MediumGroup=row['MediumGroup']
    age=row['Age']

    if sex == 1:
        if pclass3 == 1 and LargeGroup == 1:
            result['Survived'][index] = 0
            allReadyPredicted.add(index)
        elif pclass3 == 1 and MediumGroup == 1:
            if age<0.4:
                result['Survived'][index] = 1
                allReadyPredicted.add(index)
            else:
                result['Survived'][index] = 0
                allReadyPredicted.add(index)
        else:
            result['Survived'][index] = 1
            allReadyPredicted.add(index)

    else:
        result['Survived'][index] = 0
        if pclass1==1 and age<0.20:
            result['Survived'][index] = 1
            allReadyPredicted.add(index)
        else:
            result['Survived'][index] = 0
            allReadyPredicted.add(index)


result.to_csv('Data/Output/OwnPrediction.csv', header='PassengerId\tSurvived', sep=',')
print("Allready Predicted %s" % (len(allReadyPredicted)))
print("Not Predicted %s" % (test.shape[0] - len(allReadyPredicted)))
