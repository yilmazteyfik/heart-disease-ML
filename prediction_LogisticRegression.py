import numpy as np
import pandas as pd
import sklearn
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


#import csv file
heart_dataset = pd.read_csv("heart.csv")
#print first 5 rows of dataset
heart_dataset.head()
#print last 5 rows of dataset
heart_dataset.tail()
#getting info
heart_dataset.info()
#statistical measures
heart_dataset.describe()    
#check the distribution of target variable
heart_dataset["target"].value_counts()
#splitting the features and target
X = heart_dataset.drop(columns='target', axis=1)
Y = heart_dataset['target']
print(X)

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
#20 percent of data is used for testing
print(Y.shape, Y_train.shape, Y_test.shape)

#model training
model = LogisticRegression()
model.fit(X_train, Y_train)

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)

#building a predictive system
input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
#changing input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print("The person does not have heart disease")
else:
    print("The person has heart disease")  