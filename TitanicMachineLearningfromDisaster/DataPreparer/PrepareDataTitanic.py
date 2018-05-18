import numpy
import pandas
from TitanicMachineLearningfromDisaster.DataPreparer import PrepareData, Group2, PrepareGroupInformations
from TitanicMachineLearningfromDisaster.DataWorker import FilenameBuilder


class PrepareDataTitanic:
    filenameExtension = []
    preparer = PrepareData.PrepareData()
    preparerGroupInformations = PrepareGroupInformations.PrepareGroupInformations()

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
        ageMissing = numpy.zeros(dataset.shape[0])

        for index, row in missingValues.iterrows():
            similar = self.similarRows(dataset, pclass=row['Pclass'], sibsp=row['SibSp'], parch=row['Parch'])
            similarMean = dataset[similar]['Age'].median()
            ageMissing[index - 1] = 1
            if numpy.isnan(similarMean):
                dataset['Age'].ix[index] = mean
            else:
                dataset['Age'].ix[index] = similarMean

        dataset['AgeMissing'] = ageMissing
        self.filenameExtension += 'C'

    def similarRows(self, dataset, pclass, sibsp, parch):
        similar = (dataset['Pclass'] == pclass) & (dataset['SibSp'] == sibsp) & (dataset['Parch'] == parch)
        return similar

    def logScaleFare(self, dataset):
        dataset['Fare'] = dataset['Fare'].map(lambda x: numpy.log(x) if x > 0 else 0)
        self.filenameExtension += 'E'

    def scaleFareByFamilySize(self, dataset):
        dataset['FareFamilysize'] = dataset['Fare'].divide(dataset['FamilySize'])
        self.filenameExtension += 'K'

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

    def addFeatureFamily(self, dataset):
        familysize = dataset['SibSp'] + dataset['Parch'] + 1;

        dataset['Single'] = familysize.map(lambda s: 1 if s == 1 else 0)
        dataset['SmallFamily'] = familysize.map(lambda s: 1 if s == 2 else 0)
        dataset['MediumFamily'] = familysize.map(lambda s: 1 if 2 < s <= 4 else 0)
        dataset['LargeFamily'] = familysize.map(lambda s: 1 if s >= 5 else 0)
        self.filenameExtension += 'G'

    def addFeatureFamilySize(self, dataset):
        familysize = dataset['SibSp'] + dataset['Parch'] + 1;
        dataset['FamilySize'] = familysize
        self.filenameExtension += 'J'

    def addFeatureFare(self, dataset):
        fares = dataset['Fare']
        dataset['CheapFare'] = fares.map(lambda s: 1 if s <= 10 else 0)
        dataset['MediumFare'] = fares.map(lambda s: 1 if 10 < s <= 35 else 0)
        dataset['HighFare'] = fares.map(lambda s: 1 if 35 < s <= 70 else 0)
        dataset['VeryHighFare'] = fares.map(lambda s: 1 if s > 70 else 0)
        self.filenameExtension += 'V'

    def prepareCabin(self, dataset):
        dataset['Cabin'] = dataset['Cabin'].fillna('X')
        dataset['Cabin'] = dataset['Cabin'].map(lambda s: s[0])
        self.filenameExtension += 'H'

    def preparePclassSex(self, dataset):
        men12MissingAge = numpy.zeros(dataset.shape[0])
        men1Young = numpy.zeros(dataset.shape[0])
        men1Middle = numpy.zeros(dataset.shape[0])
        men1Old = numpy.zeros(dataset.shape[0])
        men2Young = numpy.zeros(dataset.shape[0])
        men2Old = numpy.zeros(dataset.shape[0])
        men3 = numpy.zeros(dataset.shape[0])
        men3Title = numpy.zeros(dataset.shape[0])

        women1 = numpy.zeros(dataset.shape[0])
        women2 = numpy.zeros(dataset.shape[0])
        women3 = numpy.zeros(dataset.shape[0])

        for index, row in dataset.iterrows():
            if (row['Sex'] == 0):
                if row['Pclass_1'] == 1:
                    if row['AgeMissing'] == 1:
                        men12MissingAge[index - 1] = 1
                    elif row['Age'] <= 14:
                        men1Young[index - 1] = 1
                    elif row['Age'] <= 35:
                        men1Middle[index - 1] = 1
                    else:
                        men1Old[index - 1] = 1
                elif row['Pclass_2'] == 1:
                    if row['AgeMissing'] == 1:
                        men12MissingAge[index - 1] = 1
                    elif row['Age'] <= 14:
                        men2Young[index - 1] = 1
                    else:
                        men2Old[index - 1] = 1
                elif row['Pclass_3'] == 1:
                    men3[index - 1] = 1
            else:
                if row['Pclass_1'] == 1:
                    women1[index - 1] = 1
                elif row['Pclass_2'] == 1:
                    women2[index - 1] = 1
                elif row['Pclass_3'] == 1:
                    women3[index - 1] = 1
        dataset['MenPc12AgeMissing'] = men12MissingAge
        dataset['MenPc1Young'] = men1Young
        dataset['MenPc1Middle'] = men1Middle
        dataset['MenPc1Old'] = men1Old
        dataset['MenPc2Young'] = men2Young
        dataset['MenPc2Old'] = men2Old
        dataset['MenPc3'] = men3
        dataset['Womenc1'] = women1
        dataset['Womenc2'] = women2
        dataset['Womenc3'] = women3
        self.filenameExtension += 'Y'

    def prepareTicket(self, dataset):
        tickets = []
        for ticket in dataset['Ticket']:
            t = ticket.split(" ")[0]
            t = t.replace("/", "")
            t = t.replace(".", "")

            if t.isdigit():
                t = 'X'
            tickets.append(t)

        dataset['TicketPrepared'] = tickets
        self.filenameExtension += 'I'

    def fareDivideByNumOfTicketnumbers(self, data):
        fares = data['Fare']
        ticketnumbers = data.groupby('Ticketnumber')['Ticketnumber'].count()

        for index, row in data.iterrows():
            ticketnumber = row['Ticketnumber']
            count = ticketnumbers.get(ticketnumber)
            fares[index] = numpy.divide(fares[index], count)

        data['Fare'] = fares
        self.filenameExtension += 'R'

    def getFileExtension(self):
        extensionTitanic = self.filenameExtension
        extensionOther = self.preparer.filenameExtension
        extensionGroup = self.preparerGroupInformations.filenameExtension
        extension = extensionTitanic
        extension.extend(extensionOther)
        extension.extend(extensionGroup)
        extension.sort()
        extensionName = ''.join(extension)

        return '%s' % (extensionName)

    def extractTicketIndicator(self, dataset):
        dataset = pandas.get_dummies(dataset, columns=['Ticketnumber'], prefix='T')
        self.filenameExtension += 'M'
        return dataset

    def calculateTicketNumber(self, dataset):
        tickets = numpy.zeros(dataset.shape[0])
        for index, row in dataset.iterrows():
            ticket = row['Ticket']
            ticket = ticket.split(' ')
            ticket = ticket[len(ticket) - 1]
            tickets[index - 1] = ticket
        dataset['Ticketnumber'] = tickets
        self.filenameExtension += 'P'

    def prepareData(self, dataset):
        dataset[["Age", "SibSp", "Parch", "Fare"]] = dataset[["Age", "SibSp", "Parch", "Fare"]].astype(float)
        dataset = self.fillMissingValues(dataset)
        self.fillMissingAge(dataset)
        self.addFeatureFare(dataset)
        self.convertSexToNumerical(dataset)
        self.mapTitles(dataset)
        self.addFeatureFamily(dataset)
        self.addFeatureFamilySize(dataset)
        self.scaleFareByFamilySize(dataset)
        self.prepareCabin(dataset)

        self.prepareTicket(dataset)
        self.calculateTicketNumber(dataset)
        self.preparerGroupInformations.calcGroups2(dataset)
        self.fareDivideByNumOfTicketnumbers(dataset)

        # self.logScaleFare(dataset)
        # self.preparer.scaleData(dataset, ['Age', 'Fare'])

        # dataset=self.extractTicketIndicator(dataset)

        dataset = self.convertIndicatorValues(dataset, ['Embarked', 'Title', 'Pclass', 'Cabin', 'TicketPrepared'])

        self.preparePclassSex(dataset)
        # dataset = self.convertIndicatorValuesGroupId(dataset)
        self.preparerGroupInformations.addFeatureGroupSize(dataset)
        self.preparerGroupInformations.convertIndicatorValuesGroupSize(dataset)

        return dataset


def prepareData():
    path = FilenameBuilder.getRootPath()
    train = pandas.read_csv("%s/Data/Input/train.csv" % (path), index_col='PassengerId')
    test = pandas.read_csv("%s/Data/Input/test.csv" % (path), index_col='PassengerId')
    dataset = pandas.concat([train, test], axis=0)

    preparerTitanic = PrepareDataTitanic()
    dataset = preparerTitanic.prepareData(dataset)

    train = dataset[dataset['Survived'].notna()]

    test = dataset[dataset['Survived'].isna()]

    extensionTest = preparerTitanic.getFileExtension()
    train = preparerTitanic.preparer.dropOutliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
    extensionTrain = preparerTitanic.getFileExtension()

    test.to_csv('%s/Data/Input/PreparedData/%s/PreparedTest_%s.csv' % (path, extensionTrain, extensionTest),
                header=True, sep=',')

    train.to_csv('%s/Data/Input/PreparedData/%s/PreparedTrain_%s.csv' % (path, extensionTrain, extensionTrain),
                 header=True, sep=',')


prepareData()
