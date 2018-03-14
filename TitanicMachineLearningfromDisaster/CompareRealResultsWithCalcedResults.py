import numpy
import pandas

def printDifferencesIncorrectCorrect(incorrectPassenger,correctPassenger):
    print("Pclass")
    print(incorrectPassenger['Pclass'].mean())
    print(correctPassenger['Pclass'].mean())
    print("Sex")

    sumMaleIncor = len(incorrectPassenger[incorrectPassenger['Sex'] == 'male'].count())
    sumIncor = len(incorrectPassenger)
    # print(sumIncor)
    print(sumMaleIncor / sumIncor)
    sumMaleCor = len(correctPassenger[correctPassenger['Sex'] == 'male'].count())
    sumCor = len(correctPassenger)
    print(sumMaleCor / sumCor)
    # print(incorrectPassenger['Sex'].mean())
    # print(correctPassenger['Sex'].mean())
    print("Age")
    print(incorrectPassenger['Age'].mean())
    print(correctPassenger['Age'].mean())
    print("SibSp")
    print(incorrectPassenger['SibSp'].mean())
    print(correctPassenger['SibSp'].mean())
    print("Parch")
    print(incorrectPassenger['Parch'].mean())
    print(correctPassenger['Parch'].mean())
    print("Fare")
    print(incorrectPassenger['Fare'].mean())
    print(correctPassenger['Fare'].mean())

realResults= pandas.read_csv('Realresults.csv',delimiter='\t')
calcedResults= pandas.read_csv('Calcedresults.csv',delimiter=',')
#print(realResults)
#print(calcedResults)
#print(type(realResults))
#print(type(calcedResults))

#print(realResults)
countStatusUnclear=0;
countIncorrectCalculation=0;
incorrectPrediction=[];
correctPrediction=[];
for index, row in realResults.iterrows():
    #s=passenger['Survived']
    status=row['Survived']
    passenger = calcedResults.ix[index]
    # print(passenger)
    statusCalced = passenger['Survived']
    if status==-1:
        countStatusUnclear=countStatusUnclear+1;
    elif status!=statusCalced:
        countIncorrectCalculation=countIncorrectCalculation+1;
        #print(row['PassengerId'],status, statusCalced)
        incorrectPrediction.append(row['PassengerId'])
    else:
        correctPrediction.append(row['PassengerId'])
    #print(statusCalced)



print(countStatusUnclear)
print("Incorrect Calculation:")
print(countIncorrectCalculation)
#print(incorrectPrediction)
#print(correctPrediction)

incorrectPassenger=pandas.DataFrame(columns=["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]);
correctPassenger=pandas.DataFrame();
testData= pandas.read_csv('test_with_corrections.csv',delimiter=',')

for index, row in testData.iterrows():
    paId = row['PassengerId']
    if paId in incorrectPrediction:
        #print(paId)
        incorrectPassenger=incorrectPassenger.append(row)
    else:
        correctPassenger=correctPassenger.append(row)
    #print(paId)

printDifferencesIncorrectCorrect(incorrectPassenger,correctPassenger)
#print(incorrectPassenger)
#print(correctPassenger)

