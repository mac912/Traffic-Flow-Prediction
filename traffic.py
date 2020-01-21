from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
from sklearn.pipeline import Pipeline

#a = np.array([(1, 2, 3, 4, 5, 6, 8), (5, 8, 9, 7, 6, 2, 8)])

#print(a.shape)

#s = range(1000)
#SIZE = 100000

#L1 = range(SIZE)
#L2 = range(SIZE)

#a1 = np.arange(SIZE)
#a2 = np.arange(SIZE)

#start = time.time()

#result = [(x, y) for x, y in  zip(L1, L2)]
#print((time.time()-start)*1000)

#start = time.time()
#result = a1+a2
#print((time.time()-start)*1000)

#print(sys.getsizeof(5)*len(s))

#D = np.arange(1000)
#print(D.size*D.itemsize)

#with open('test.csv', 'r') as csv_file:
#    csv_reader = csv.DictReader(csv_file)

#    for line in csv_reader:
#        print(line)


data = pd.read_csv('Book2.csv')
df = pd.DataFrame(data)
df2 = df.tail(30)
print(len(df))
#df2 = df.tail(0)
df=df[['day', 'traffic', 'Time']].head(len(df)-30)
x= np.array(df[['day', 'Time']])
y=np.array(df['traffic'])
#print(df.tail(5))
#print(df2.tail(5))
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.1)

#decisionclassifier = DecisionTreeClassifier()
#decisionclassifier.fit(x_train, y_train)
#svclassifier = SVC(kernel='poly', degree=8)
#svclassifier.fit(x_train, y_train)
#y_pred = svclassifier.predict(x_test)
#acc=decisionclassifier.score(x_test, y_test)
decisionclassifier = DecisionTreeClassifier()
decisionclassifier.fit(x_train, y_train)
y1_pred = decisionclassifier.predict(x_test)
acc=decisionclassifier.score(x_test, y_test)
t=0
r =0
for m, n in zip(y1_pred, y_test):
    p = m-n
    if(p<0):
        p=-1*p
        d = (p*100)/n
        r = r+d
        t = t+1
    else:
        c = (p*100)/m
        r = r+c
        t = t+1
z1 = 100-(r/t)
print("decision tree percent", z1)


nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(x_train, y_train) 
y2_pred = nca_pipe.predict(x_test)
#print(y2_pred)
#print(y_pred)
#print(y_test)
#print(nca_pipe.score(x_test, y_test)) 
#print(y_pred)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#predict = mlr.predict(df2[['day', 'lane', 'Time']])
#print(y_pred)
#print(y_test)
t=0
r =0
for m, n in zip(y2_pred, y_test):
    p = m-n
    if(p<0):
        p=-1*p
        d = (p*100)/n
        r = r+d
        t = t+1
    else:
        c = (p*100)/m
        r = r+c
        t = t+1
z2 = 100-(r/t)
print("knn percent", z2)

mlr=LinearRegression()
mlr.fit(x_train,y_train)
mlr_accu=mlr.score(x_test,y_test)
predict = mlr.predict(df2[['day', 'Time']])
df2['pred_mlr'] = [i for i in predict]
#print(df2[['pred_mlr','traffic']])

#print(y3_pred)
t=0
r =0
for m, n in zip(df2['pred_mlr'], df2['traffic']):
    p = m-n
    if(p<0):
        p=-1*p
        d = (p*100)/n
        r = r+d
        t = t+1
    else:
        c = (p*100)/m
        r = r+c
        t = t+1
z3 = 100-(r/t)
print("mlr percent", z3)

clf = SVC(gamma='auto')
clf.fit(x_train, y_train) 
y3_pred = clf.predict(x_test)
t=0
r =0
for m, n in zip(y3_pred, y_test):
    p = m-n
    if(p<0):
        p=-1*p
        d = (p*100)/n
        r = r+d
        t = t+1
    else:
        c = (p*100)/m
        r = r+c
        t = t+1
z4 = 100-(r/t)
print("svm percent", z4)
print("Actual value",",", "decison tree",",", "knn",",", "svm",",", "mlr")
for a1, a2, a3, a4, a5 in zip(y_test, y1_pred, y2_pred, y3_pred, df2['pred_mlr']):
    print(a1, end=" ")
    print(a2, end=" ")
    print(a3, end=" ")
    print(a4, end=" ")
    print(a5)
data = [z3, z1, z4, z2]
register = 0, 1, 2, 3
algo = "mlr", "decision_tree", "svm", "nearest_neighbour"
plt.figure(figsize=(8, 4))
plt.bar(register, data, width=0.8, color=("m", "r", "g", "b"))
plt.title("ALGO", fontsize=20)
plt.xticks(register, algo)
plt.show()

#time = [1, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
#pred = [44.91, 58.09, 78.07, 107.7, 138.5, 170.6]
#traffics = [449.48, 553.57, 696.783, 870.133, 1000.
#plt.plot(time, y3_pred, color='g')
#plt.plot(time, y_test, color='orange')
#plt.xlabel('Time')
#plt.ylabel('traffic')
#plt.title('Pakistan India Population till 2010')
#plt.show()

#sum=0
#for m, n in zip(y_pred, y_test):
#    p = m-n
#    if(p<0):
#        p = -1*p
    #print(p, end=" ")
#    sum = sum+p
#r = sum/30
#df2['y_pred'] = [i for i in y_pred]
#print(df2[['y_pred','traffic']])

#print(mlr_accu)
#ar = np.array(x)
#for i in range(1, 4320):
#    print(ar[i].split(' ')[1])

#ss= data.split('\n')
#print()


#y = df['Flow'].values
#y1 = np.array(y)
#reg = LinearRegression().fit(x, y1)
#reg.score(x, y1)

#y = df['Lane Flow'].values

#y = df['Lane 1 flow'].values
#print(x)
#print(df[['5 Minutes']])
#print(df.loc["04/03/2016 0:05", "10"])

    # with open('file.csv', 'w') as new_file:
    #    csv_writer = csv.writer(new_file)
    #    for line in csv_reader:
    #        csv_writer.writerow(line)
#now = datetime.now()
#print(now)
#timestamp = datetime.timestamp(now)
#print("timestamp =", timestamp)
