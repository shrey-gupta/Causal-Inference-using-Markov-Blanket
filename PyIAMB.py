from __future__ import division
import pandas as pd
from pandas import DataFrame
import codecs,json
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import numpy as np
import random
from sklearn.metrics import mutual_info_score
import pandas as pd
import entropy_estimators as ee
from sklearn import preprocessing

###############################################################################################
####THIS BLOCK OF CODE IS FOR LUCAS DATSET INITIALIZATION#############

'''df1 = DataFrame()
df1 = pd.read_csv("LUCASDATA/lucas0_train.csv",delim_whitespace=False,header=0)
df1 = df1.iloc[:,:-1]


df2 = DataFrame()
df2 = pd.read_csv("LUCASDATA/lucas0_train_targets.csv",delim_whitespace=False,header=0)'''

################################################################################################
############### THIS BLOCK OF CODE IS FOR THE OTHER DATASETS ###################################

f = open("REGED/columnname","w")
for i in range(1,1000):           ##### RANGE FOR THE DATASET
    f.write("A"+str(i)+" ")

f.write('\n')
f.close()

fin = open("REGED/reged0_train.data", "r")
data2 = fin.read()
fout = open("REGED/columnname", "a")

for i in data2:
    fout.write(i)

fout.close()
fin.close()
df1 = pd.read_csv("REGED/columnname",delimiter=" ")
df1 = df1.iloc[:,:-1]
df3 = DataFrame()

df3 = df1
df3 = df3.values
X = np.array(df3)
X_scaled = preprocessing.scale(X)

index = [df1.columns]
#print(index)

df1 = pd.DataFrame(data = X_scaled, columns = index)
df2 = DataFrame()
df2 = pd.read_csv("REGED/reged0_train_targets.csv",delim_whitespace=False,header=0)

########################################################################################################

V = set(df1.columns) ## Constant set
T = set(df2.columns) ## Constant set

## New sets
CMB = set() ## Constant set
Temp = set() ## Temporary set
CMBTemp = set() ## Temporary set


############################## GROWING PHASE ############################################################

oldLen = len(CMB)
newLen = len(CMB)


Threshold = 0.01

while True:
    Temp.update(V)
    max = 0

    for alpha in range(len(Temp)):
    	ele = Temp.pop()

    	l1 = df1[ele].tolist()
    	l2 = df2['Z'].tolist()
    	x=[]
    	for i in l1:
    		x.append(i)
    	y=[]
    	for j in l2:
    		y.append(j)

    	CMBTemp.update(CMB)
    	CMScore = 1
    	hmax = 0

        #CMBTemp means Conditional Markov Blanket Temporary Set.
    	if len(CMBTemp)==0:
            CMScore = ee.midd(x,y)

        else :
            #You can replace this whole with Conditional Mutual Information function. The score that you get will be the Half_Score
            for q in range(len(CMBTemp)):
                condele = CMBTemp.pop()
                l3 = df1[condele].tolist()
                z=[]
                for k in l3:
                    z.append(k)
                ent = ee.entropyd(z)
                MI1 = ee.midd(x,z)
                MI2 = ee.midd(y,z)
                Half_Score = (MI1/ent)*MI2

                if Half_Score > hmax :
                    hmax = Half_Score

        #print "CMI for feature is :",CMScore
        CMScore = (ee.midd(y,x) - hmax)

        #Initially max is 0
        if CMScore > max:
        	max = CMScore
        	maxInd = ele


    if max > Threshold:
        CMB.add(maxInd)
        V.discard(maxInd)
        print("The max feature is :",maxInd)
        print(" The score is :",max)
        newLen = newLen + 1

    #look into this newLen and oldLen thing. What is it's purpose and how can we replace it.
    #Right now, it represents features being added to CMB and hence, it's length getting increased.
    #But why do we increase the oldLen too.
    if (newLen - oldLen) < 1:
        print("NO MORE FEATURES ADDED !!")
        break
    else:
        oldLen = oldLen + 1

print("The Current Markov Blanket for the dataset is :",CMB)



####################################### SHRINKING PHASE ######################################################

CMBTemp = set()
min = 0
CMScore = 1

tempDict = {}
list4 = []

'''for e1 in range(0,len(CMB)):
	t3ele = CMB.pop()
	list3.append(t3ele)
	CMBTemp.add(t3ele)

CMB = set(list3)'''

CMBTemp.update(CMB)

for alpha in range(len(CMBTemp)):
	NCMBTemp = set()
	CMScore = 1
	ele = CMBTemp.pop()

	l1 = df1[ele].tolist()
	l2 = df2['Z'].tolist()
	x=[]
	for i in l1:
		x.append(i)
	y=[]
	for i in l2:
		y.append(i)

	CMB.discard(ele)
	NCMBTemp.update(CMB)

	hmax = 0

	if len(NCMBTemp)==0:
		CMScore = ee.midd(x,y)

	else :
		for q in range(len(NCMBTemp)):
			condele = NCMBTemp.pop()
			l3 = df1[condele].tolist()
			z=[]
			for i in l3:
				z.append(i)
			ent = ee.entropyd(z)
			MI1 = ee.midd(x,z)
			MI2 = ee.midd(y,z)
			#print Score
			Half_Score = (MI1/ent)*MI2

			if Half_Score > hmax :
				hmax = Half_Score


	CMScore = (ee.midd(y,x) - hmax)

	print("CMI for feature ",ele,"is :",CMScore)
	CMB.add(ele)
	tempDict[ele] = CMScore


c =1

for item in tempDict.keys():
	minVal = tempDict.get(item)
	if minVal < Threshold :
		CMB.discard(item)

print("The Final Markov Blanket for the dataset is :",CMB)

####################################### Testing for Accuracy.###############################################

from sklearn import svm

dfTest = DataFrame()

#############################################################################################################
############################### THIS BLOCK IS FOR LUCAS DATASET #############################################

'''dfTest = pd.read_csv("LUCASDATA/lucas0_test.csv",delim_whitespace=False,header=0)
dfTest = dfTest.iloc[:,:-1]'''

#############################################################################################################
############################### THIS BLOCK IS FOR THE REMAINING DATASETS ####################################
f = open("REGED/columnname2","w")

for i in range(1,1000):             #### RANGE FOR THE DATASET
    f.write("A"+str(i)+" ")

f.write('\n')
f.close()

fin = open("REGED/reged0_test.data", "r")
data2 = fin.read()

fout = open("REGED/columnname2", "a")

for i in data2:
    fout.write(i)

fout.close()
fin.close()

dfTest =pd.read_csv("REGED/columnname2",delimiter=" ")

dfTest = dfTest.iloc[:,:-1]

df6 = DataFrame()

df6 = dfTest
df6 = df6.values

X = np.array(df6)
X_scaled = preprocessing.scale(X)

index = [dfTest.columns]

#print(index)

dfTest = pd.DataFrame(data = X_scaled, columns = index)
##################################################################################################################

X = np.array(df1)
y = np.array(df2)
clf = svm.SVC(C=0.7, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=2, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X, y)

Z = np.array(dfTest)

test_Target = clf.predict(Z)
print(test_Target)

#################################################################################################################
df4 = DataFrame()
DCMB = set()
DCMB.update(CMB)


setEle = DCMB.pop()
df4 = df1[setEle]


for q in range(len(DCMB)):
	setEle = DCMB.pop()
	tempSet = set()
	tempSet = df1[setEle]
	df4 = pd.concat([df4,tempSet],axis=1)
#print df4

X1 = np.array(df4)
y1 = np.array(df2)
clf1 = svm.SVC(C=0.7, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=2, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf1.fit(X1, y1)


####################################################################################################################

df5 = DataFrame()
DCMB2 = set()
DCMB2.update(CMB)


setEle = DCMB2.pop()
df5 = dfTest[setEle]


for q in range(len(DCMB2)):
	setEle = DCMB2.pop()
	tempSet = set()
	tempSet = dfTest[setEle]
	df5 = pd.concat([df5,tempSet],axis=1)
#print df4

Z1 = np.array(df5)
final_predict = clf1.predict(Z1)

acc1 = np.sum(final_predict == test_Target) #/len(test_Target))*100
acc2 = len(test_Target)

print("The accuracy of the model for the dataset is: ",(acc1/acc2)*100,"%")
