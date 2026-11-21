import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split

#Data Importing
data = pd.read_csv('C:/Users/india/OneDrive/Desktop/AI and ML/car.data')

#tells to access which coloumns to access from data
x = data[['buying', 'maint', 'safety']]
y = data[['class']]  

#converting data into ints as KNN 
#y:
y_label_mapping = {'unacc':0, 'acc':1,'good':2,'vgood':3}
y = data[['class']].replace(y_label_mapping).astype(int)

#x
x_label_mapping = {'low': 0,'med': 1, 'high': 2, 'vhigh': 3 }
x = data[['buying', 'maint', 'safety']].replace(x_label_mapping).astype(int)

#CREATING MODEL 
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform') #n_neighors s same as k
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn.fit(x_train,y_train)

predictions = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test,predictions)
print("predictions: ", predictions)
print("accuracy",accuracy)

# Accuracy keeps changing @ each run beacuse test_train_split splits data into testing and training randomly each time to stop it specify random_state = 42

mapping = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
class_names = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}

buying = input("buying (low/med/high/vhigh): ")
maint  = input("maint (low/med/high/vhigh): ")
safety = input("safety (low/med/high): ")

result = knn.predict([[mapping[buying], mapping[maint], mapping[safety]]])
print("Prediction:", class_names[result[0]])
