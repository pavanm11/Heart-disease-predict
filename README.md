# Heart-disease-prediction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
data = pd.read_csv('heart.csv')
data.head()
data.tail()
data.shape
data.info()
data.isnull().sum()
data.describe()
data['target'].value_counts()
X = data.drop(columns='target',axis=1)
Y =data['target']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
model = LogisticRegression()
model.fit(X_train,Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data : ',training_data_accuracy)
X_test_predictionn = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predictionn,Y_test)
print('Accuracy on test data : ',test_data_accuracy)
input_data = (67,1,0,100,299,0,0,125,1,0.9,1,2,2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
  print('the person does not have heart disease')
else:
  print('the person has heart disease')
