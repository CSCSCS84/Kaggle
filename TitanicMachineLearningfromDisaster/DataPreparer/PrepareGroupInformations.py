from TitanicMachineLearningfromDisaster.DataPreparer import  Group2
import numpy
import pandas

class PrepareGroupInformations:
    filenameExtension = []

    def addFeatureGroupSize(self, dataset):
        groupSize = dataset['GroupSize']

        dataset['SingleGroup'] = groupSize.map(lambda s: 1 if s == 1 else 0)
        dataset['SmallGroup'] = groupSize.map(lambda s: 1 if s == 2 else 0)
        dataset['MediumGroup'] = groupSize.map(lambda s: 1 if 2 < s <= 5 else 0)
        dataset['LargeGroup'] = groupSize.map(lambda s: 1 if s >= 6 else 0)
        self.filenameExtension += 'U'

    def calcGroups2(self, dataset):
        groups = self.findGroups2(dataset)
        groupIds = numpy.zeros((dataset.shape[0], 1))
        numOfDeads = numpy.zeros((dataset.shape[0], 1))
        numOfSurvived = numpy.zeros((dataset.shape[0], 1))
        groupSize= numpy.zeros((dataset.shape[0], 1))
        familySurvival=numpy.zeros((dataset.shape[0], 1))
        for group in groups:
            passengers = group.passengers
            for p in passengers:
                id = p['PId']
                # groupIds[id]=group.id
                groupIds[id - 1] = group.id
                row = dataset.ix[id]
                parch = row['Parch']
                sibSb = row['SibSp']

                if parch + sibSb + 1 > len(group.passengers):
                    groupSize[id - 1]=parch + sibSb + 1
                else:
                    groupSize[id-1]=len(group.passengers)

                if p['Survived'] == 0:
                    numOfDeads[id - 1] = group.numOfDead - 1
                else:
                    numOfDeads[id - 1] = group.numOfDead
                if p['Survived'] == 1:
                    numOfSurvived[id - 1] = group.numOfSurvived - 1
                else:
                    numOfSurvived[id - 1] = group.numOfSurvived

                if numOfSurvived[id - 1] >=1:
                    familySurvival[id-1]=1
                elif numOfDeads[id - 1] >=1:
                    familySurvival[id - 1] = 0
                else:
                    familySurvival[id - 1] = 0.5

        dataset['GroupId'] = groupIds
        dataset['GroupDead'] = numOfDeads
        dataset['GroupSurvived'] = numOfSurvived
        dataset['GroupDeadRatio2'] = dataset['GroupDead'].divide(dataset['GroupDead'] + dataset['GroupSurvived'])
        dataset['GroupSurvivedRatio2'] = dataset['GroupSurvived'].divide(
            dataset['GroupDead'] + dataset['GroupSurvived'])
        dataset['GroupDeadRatio2'] = dataset['GroupDeadRatio2'].fillna(0.0)
        dataset['GroupSurvivedRatio2'] = dataset['GroupSurvivedRatio2'].fillna(0.0)
        dataset['GroupSize']=groupSize
        dataset['GroupSurvival']=familySurvival
        self.filenameExtension += 'QQ'

    # group2 procedure
    def findGroups2(self, dataset):
        groups = []
        for index, row in dataset.iterrows():
            group = Group2.Group2(row['Ticketnumber'], row['Fare'], -1, row['Pclass'])
            groupId = self.findGroup2(groups, group)
            row['PId'] = index
            # print(index)
            if groupId == -1:
                group.id = len(groups)
                group.add(row)

                if row['Survived'] == 0:
                    group.numOfDead += 1
                elif row['Survived'] == 1:
                    group.numOfSurvived += 1
                groups.insert(len(groups), (group))
            else:
                # groups[groupId].add(row)
                if row['Survived'] == 0:
                    groups[groupId].numOfDead += 1
                elif row['Survived'] == 1:
                    groups[groupId].numOfSurvived += 1
                groups[groupId].passengers.insert(len(groups[groupId].passengers), row)

        return groups

    # group2 procedure
    def findGroup2(self, groups, group):

        for g in groups:
            if group == g:
                return g.id
        return -1

    def convertIndicatorValuesGroupId(self, dataset):
        dataset = pandas.get_dummies(dataset, columns=['GroupId'], prefix=['GroupId'])
        self.filenameExtension += 'S'
        return dataset

    def convertIndicatorValuesGroupSize(self, dataset):
        groupSize=dataset['GroupSize']
        dataset = pandas.get_dummies(dataset, columns=['GroupSize'], prefix=['GroupSize'])
        dataset['GroupSize']=groupSize
        self.filenameExtension += 'T'
        return dataset