import pandas as pd
import numpy as np


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


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

# Predict the Test set results
y_pred = regressor.predict(X_test)
score = regressor.score(X_test,y_test)

