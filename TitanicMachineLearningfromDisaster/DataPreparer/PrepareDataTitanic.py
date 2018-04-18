import numpy
import pandas
from TitanicMachineLearningfromDisaster import PrepareData


class PrepareDataTitanic:
    filenameExtension = []
    preparer = PrepareData.PrepareData()

    def convertSexToNumerical(self, dataset):
        dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
        self.filenameExtension += 'A'

    def convertIndicatorValues(self, dataset, features):
        for f in features:
            dataset = pandas.get_dummies(dataset, columns=[f], prefix=f)
        self.filenameExtension += 'B'
        return dataset

    def fillMissingValues(self, dataset):
        dataset = dataset.fillna(numpy.nan)
        dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        return dataset

    def fillMissingAge(self, dataset):
        missingValues = dataset[dataset['Age'].isnull()];
        mean = dataset['Age'].median()

        for index, row in missingValues.iterrows():
            similar = self.similarRows(dataset, pclass=row['Pclass'], sibsp=row['SibSp'], parch=row['Parch'])
            similarMean = dataset[similar]['Age'].median();
            if numpy.isnan(similarMean):
                dataset['Age'].ix[index] = mean;
            else:
                dataset['Age'].ix[index] = similarMean
        self.filenameExtension += 'C'

    def similarRows(self, dataset, pclass, sibsp, parch):
        similar = (dataset['Pclass'] == pclass) & (dataset['SibSp'] == sibsp) & (dataset['Parch'] == parch)
        return similar

    def logScaleFare(self, dataset):
        dataset['Fare'] = dataset['Fare'].map(lambda x: numpy.log(x) if x > 0 else 0)
        self.filenameExtension += 'E'

    def mapTitles(self, dataset):
        surnames = [i.split(',')[1] for i in dataset['Name']]
        titles = [i.split('.')[0].strip() for i in surnames]
        dataset['Title'] = titles;

        dataset["Title"] = dataset["Title"].map(
            {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2,
             'Lady': 3, 'the Countess': 3, 'Countess': 3, 'Capt': 3, 'Col': 3, 'Don': 3,
             'Dr': 3, 'Major': 3,
             'Rev': 3, 'Sir': 3, 'Jonkheer': 3, 'Dona': 3})
        self.filenameExtension += 'F'

    def addFeatureFamilySize(self, dataset):
        familysize = dataset['SibSp'] + dataset['Parch'] + 1;

        dataset['Single'] = familysize.map(lambda s: 1 if s == 1 else 0)
        dataset['SmallFamily'] = familysize.map(lambda s: 1 if s == 2 else 0)
        dataset['MediumFamily'] = familysize.map(lambda s: 1 if 2 < s <= 4 else 0)
        dataset['LargeFamily'] = familysize.map(lambda s: 1 if s >= 5 else 0)
        self.filenameExtension += 'G'

    def prepareCabin(self, dataset):
        dataset['Cabin'] = dataset['Cabin'].fillna('X')
        dataset['Cabin'] = dataset['Cabin'].map(lambda s: s[0])
        self.filenameExtension += 'H'

    def prepareTicket(self, dataset):
        tickets = []
        for ticket in dataset['Ticket']:
            t = ticket.split(" ")[0]
            t = t.replace("/", "")
            t = t.replace(".", "")

            if t.isdigit():
                t = 'X'
            tickets.append(t)

        dataset['Ticket'] = tickets
        self.filenameExtension += 'I'

    def prepareData(self, dataset):
        dataset[["Age", "SibSp", "Parch", "Fare"]] = dataset[["Age", "SibSp", "Parch", "Fare"]].astype(float)
        dataset = self.fillMissingValues(dataset)
        self.fillMissingAge(dataset)


        self.logScaleFare(dataset)
        self.preparer.scaleData(dataset,['Age','Fare'])


        self.convertSexToNumerical(dataset)
        self.mapTitles(dataset)
        self.addFeatureFamilySize(dataset)
        self.prepareCabin(dataset)
        self.prepareTicket(dataset)

        dataset = self.convertIndicatorValues(dataset, ['Embarked', 'Title', 'Pclass', 'Cabin', 'Ticket'])
        return dataset

    def fileExtension(self):
        self.filenameExtension.sort()
        extensionTitanic = ''.join(self.filenameExtension)

        self.preparer.filenameExtension.sort()
        extension = ''.join(self.preparer.filenameExtension)

        return '%s%s' % (extensionTitanic, extension)


train= pandas.read_csv("Data/Input/test.csv", index_col='PassengerId')
test = pandas.read_csv("Data/Input/train.csv", index_col='PassengerId')
dataset = pandas.concat([train, test], axis=0)

preparerTitanic = PrepareDataTitanic()
dataset = preparerTitanic.prepareData(dataset)

train = dataset[dataset['Survived'].notna()]

test = dataset[dataset['Survived'].isna()]

extension = preparerTitanic.fileExtension()
test.to_csv('Data/Input/PreparedData/PreparedTest_%s.csv' % (extension), header=True, sep=',')

train=preparerTitanic.preparer.dropOutliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
extension = preparerTitanic.fileExtension()
train.to_csv('Data/Input/PreparedData/PreparedTrain_%s.csv' % (extension), header=True, sep=',')