import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import data and create numpy arrays
dataset = pd.read_csv("data/parkinsons_updrs.data",header=0)
X = dataset.iloc[:,[0,1,2,3,4,7,15,17,18,19,20,21]].values
y = dataset.iloc[:,5].values



# Encode categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X1 = sc_X.fit_transform(X_train)

y1 = np.squeeze(sc_y.fit_transform(y_train.reshape(-1, 1)))


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X1, y1)


for k in ['linear','poly','rbf','sigmoid']:
    clf = SVR(kernel=k)
    clf.fit(X1, y1)
    confidence = clf.score(X1, y1)
    print(k,confidence)

# Predict the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.fit_transform(X_test)))
score = regressor.score(sc_X.fit_transform(X_test),sc_y.fit_transform(y_test.reshape(-1, 1)))

from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)

