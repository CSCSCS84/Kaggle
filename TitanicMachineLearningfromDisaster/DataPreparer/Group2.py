class Group2:
    def __init__(self, ticketnumber=None, fare=None, id=None, Pclass=None,numOfDead=None, numOfSurvived=None):
        if id is None:
            self.id=-1
        else:
            self.id=id

        if ticketnumber is None:
            self.ticketnumberAvg = 0
        else:

            self.ticketnumberAvg = int(ticketnumber)
        if fare is None:
            self.fareAvg = 0
        else:
            self.fareAvg = fare
        if Pclass is None:
            self.Pclass=-1
        else:
            self.Pclass=Pclass

        if numOfDead is None:
            self.numOfDead = 0
        else:
            self.numOfDead = numOfDead
        if numOfSurvived is None:
            self.numOfSurvived = 0
        else:
            self.numOfSurvived = numOfSurvived

        self.passengers=[]



    def __hash__(self):
        return hash((self.fare))

    def __eq__(self, obj):
        fares=sorted([self.fareAvg, obj.fareAvg])
        if self.ticketnumberAvg ==-1:
            return False
        elif self.ticketnumberAvg==obj.ticketnumberAvg:
            return True
        elif (abs(self.ticketnumberAvg-obj.ticketnumberAvg)<5 and (fares[0]*1.05>=fares[1]) and (self.Pclass == obj.Pclass)):
            return False
        else:
            return False

    def add(self, row):
        self.passengers.insert(len(self.passengers), (row))

    def __str__(self):
        output=''
        for p in self.passengers:
            output+='%s %s %s %s' % (p['Name'],p['Fare'],p['Ticketnumber'],self.id)
        return output
