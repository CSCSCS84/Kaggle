from TitanicMachineLearningfromDisaster.TitanicInstance import TitanicInstanceCreator
import pandas
import numpy

# from sets import Set

def notPredicted(test, predicted):

    testNotPredicted=test.drop(list(predicted))
    return testNotPredicted

fileNameExtension = 'ABCEFGHIJKPQRT12'
fileNameExtensionTest = 'ABCEFGHIJKPQRT1'
featureNumber = 2
titanic = TitanicInstanceCreator.createInstance(fileNameExtension, fileNameExtensionTest, featureNumber)

test = titanic.test
result = pandas.DataFrame(index=test.index)
result['Survived'] = numpy.ones((test.shape[0], 1)) * (-1)

allReadyPredicted = set([])

for index, row in test.iterrows():
    pclass1 = row['Pclass_1']
    sex = row['Sex']
    if pclass1 == 1 and sex == 1 and index not in allReadyPredicted:
        result['Survived'][index] = 1
        allReadyPredicted.add(index)

for index, row in test.iterrows():
    GroupSize_11 = row['GroupSize_11.0']
    GroupSize_9 = row['GroupSize_9.0']
    if (GroupSize_11 == 1 or GroupSize_9==1) and index not in allReadyPredicted:
        result['Survived'][index] = 0
        allReadyPredicted.add(index)

for index, row in test.iterrows():
    GroupSize_7 = row['GroupSize_7.0']
    GroupSize_8 = row['GroupSize_8.0']
    pclass3 = row['Pclass_3']
    if (GroupSize_7 == 1 or GroupSize_8==1 and pclass3==1) and index not in allReadyPredicted:
        result['Survived'][index] = 0
        allReadyPredicted.add(index)

##------------------men predictions------------------------

#pclass3
for index, row in test.iterrows():
    pclass3 = row['Pclass_3']
    sex = row['Sex']
    age=row['Age']
    if (sex == 0 and pclass3==1) and index not in allReadyPredicted:
        if age>0.15:
            result['Survived'][index] = 0
            allReadyPredicted.add(index)
        else:
            GroupSize_2 = row['GroupSize_2.0']
            GroupSize_3 = row['GroupSize_3.0']
            if GroupSize_2==0 and GroupSize_3==0:
                result['Survived'][index] = 0
                allReadyPredicted.add(index)
            else:
                groupDeadRatio=row['GroupDeadRatio2']
                groupSurvivedRatio = row['GroupSurvivedRatio2']
                if groupDeadRatio>groupSurvivedRatio:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)



#PClass1
for index, row in test.iterrows():
    pclass1 = row['Pclass_1']
    sex = row['Sex']
    GroupSize_1 = row['GroupSize_1.0']
    GroupSize_2 = row['GroupSize_2.0']
    GroupSize_3 = row['GroupSize_3.0']
    GroupSize_4 = row['GroupSize_4.0']
    GroupSize_5 = row['GroupSize_5.0']
    GroupSize_6 = row['GroupSize_6.0']

    groupDeadRatio = row['GroupDeadRatio2']
    groupSurvivedRatio = row['GroupSurvivedRatio2']
    age=row['Age']
    if index not in allReadyPredicted:
        if pclass1==1 and sex==0:
            if GroupSize_1==1:
                if age<=0.35:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)

            elif GroupSize_2==1:
                if age > 0.4:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)
                else:
                    if groupDeadRatio > groupSurvivedRatio:
                        result['Survived'][index] = 0
                        allReadyPredicted.add(index)
                    else:
                        result['Survived'][index] = 1
                        allReadyPredicted.add(index)

            elif GroupSize_3 == 1:
                if age < 0.44:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)
            elif GroupSize_4 == 1:
                if age < 0.44:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)

            elif GroupSize_5 == 1 or GroupSize_6 == 1:
                if age < 0.23:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)


    #if (sex == 0 and pclass1==1) and index not in allReadyPredicted:
    #    result['Survived'][index] = 0
    #    allReadyPredicted.add(index)


for index, row in test.iterrows():
    pclass2 = row['Pclass_2']
    sex = row['Sex']
    GroupSize_1 = row['GroupSize_1.0']
    GroupSize_2 = row['GroupSize_2.0']
    GroupSize_3 = row['GroupSize_3.0']
    GroupSize_4 = row['GroupSize_4.0']
    GroupSize_5 = row['GroupSize_5.0']
    GroupSize_6 = row['GroupSize_6.0']
    GroupSize_7 = row['GroupSize_7.0']

    groupDeadRatio = row['GroupDeadRatio2']
    groupSurvivedRatio = row['GroupSurvivedRatio2']
    age=row['Age']

    if index not in allReadyPredicted:
        if pclass2==1 and sex==0:
            if GroupSize_1==1 or GroupSize_2==1 or GroupSize_5==1 or GroupSize_6==1 or GroupSize_7==1:
                result['Survived'][index] = 0
                allReadyPredicted.add(index)
            else:
                if age<0.25:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)


for index, row in test.iterrows():
    pclass2 = row['Pclass_2']
    sex = row['Sex']

    if index not in allReadyPredicted:
        if pclass2==1 and sex==1:
            result['Survived'][index] = 1
            allReadyPredicted.add(index)

for index, row in test.iterrows():
    pclass3 = row['Pclass_3']
    sex = row['Sex']
    GroupSize_1 = row['GroupSize_1.0']
    GroupSize_2 = row['GroupSize_2.0']
    GroupSize_3 = row['GroupSize_3.0']
    GroupSize_4 = row['GroupSize_4.0']
    GroupSize_5 = row['GroupSize_5.0']
    GroupSize_6 = row['GroupSize_6.0']
    GroupSize_7 = row['GroupSize_7.0']
    GroupDeadRatio2 = row['GroupDeadRatio2']
    GroupSurvivedRatio2 = row['GroupSurvivedRatio2']

    if index not in allReadyPredicted:
        if pclass3 == 1 and sex == 1:
            if GroupSize_5==1 or GroupSize_6==1 or GroupSize_7==1:
                result['Survived'][index] = 0
                allReadyPredicted.add(index)
            elif GroupSize_1==1:
                if age<0.3:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)

            elif GroupSize_2 == 1:
                if GroupDeadRatio2==1.0:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)

                elif GroupSurvivedRatio2==1.0:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
            elif GroupSize_3 == 1:
                if age <0.35:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)
            elif GroupSize_4 == 1:
                if age <0.4:
                    result['Survived'][index] = 1
                    allReadyPredicted.add(index)
                else:
                    result['Survived'][index] = 0
                    allReadyPredicted.add(index)



testNotPredicted=notPredicted(test,allReadyPredicted)
print(testNotPredicted[(testNotPredicted['Sex']==1) &(testNotPredicted['Pclass_2']==1)].shape[0])

result.to_csv('Data/Output/OwnPrediction.csv', header='PassengerId\tSurvived', sep=',')
print("Allready Predicted %s" % (len(allReadyPredicted)))
print("Not Predicted %s" % (test.shape[0] - len(allReadyPredicted)))
