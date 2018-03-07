import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns


def editStringSurvivor(name):

    nameEdit = name.lower();
    nameEdit = nameEdit.replace("mr", "mr.");
    nameEdit = nameEdit.replace("mrs", "mrs.");
    nameEdit = nameEdit.replace("miss", "miss.");
    nameEdit = nameEdit.replace("master", "master.");
    nameEdit = nameEdit.replace(" ", "");

    return nameEdit;

def editStringVictims(name):
    nameEdit = name.lower();
    nameEdit = nameEdit.replace("mr", "mr.");
    nameEdit = nameEdit.replace("mrs", "mrs.");
    nameEdit = nameEdit.replace("miss", "miss.");
    nameEdit = nameEdit.replace("master", "master.");
    nameEdit = nameEdit.replace(" ", "");

    return nameEdit;

test_data = pandas.read_csv('test_with_corrections.csv',index_col='PassengerId')
f = open('survivors.csv',"r")
surivor=f.read()

surivorEdit=editStringSurvivor(surivor)
surivor = surivor.replace(" ", "");
surivor = surivor.lower();
test_data_name=test_data['Name']
#test_data_name=test_data_name.replace(" ","");
#print(test_data_name)
#print(surivor)

fi = open('victims2.csv',"r")
victims=fi.read()


victimsEdit=editStringVictims(victims)
victims=victims.replace(" ","");
victims=victims.lower();
print(victims)
print("------------------------------------------------------------")
print(victimsEdit)
print("--------------------------------------------------------------")
re = numpy.zeros(([418, 2]), dtype=int)
j = 0;
print(surivorEdit)
for name in test_data_name:

    nameCopy = name.replace(" ", "");
    nameCopy=nameCopy.lower();
    print(nameCopy)

    if nameCopy in surivor or nameCopy in surivorEdit:
        re[j][1] = 1;
        re[j][0] = j + 892;
    elif (nameCopy in victims or nameCopy in victimsEdit):
        re[j][1] = 0;
        re[j][0] = j + 892;
        #print(nameCopy)
    else:
        #print(nameCopy)
        re[j][1] = -1;
        re[j][0] = j + 892;
    j=j+1;

dresult=pandas.DataFrame(re, columns=['PassengerId','Survived'])
#print(test_data_name)
#print(dresult)
#print(type(test_data_name))
#print(dresult.dtypes)
#dresult['Name']=pandas.Series(test_data_name,index=dresult.index)
#print(test_data_name.shape)
#print(test_data_name.dtype)

result = dresult.join( test_data_name, on=['PassengerId'])
#print(result)
#print(victims)
#print(result)
print(result)
numpy.savetxt(r'Realresults.csv', result.values, fmt='%s',header='PassengerId\tSurvived\tName',delimiter='\t',comments="")