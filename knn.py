import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# import data and create numpy arrays
dataset = pd.read_csv("data/parkinsons_updrs.data",header=0)
X = dataset.iloc[:,[0,1,2,3,4,7,15,17,18,19,20,21]].values


# plot total_URDPS distributions
import seaborn as sns
sns.set(style="whitegrid")
ax = sns.boxplot(x=dataset.iloc[:,5].values)
ax1 = sns.scatterplot(x="total_UPDRS", y="sex", data=dataset)
ax2 = sns.scatterplot(x="total_UPDRS", y="age", data=dataset)

# we see that the median and mean of total_URDPS is about 30, and standard deviation is 10, so we
# set to categories of total URDPS, first over 30 and second under 30
print(dataset.iloc[:,5].std(),dataset.iloc[:,5].mean(),dataset.iloc[:,5].median())
y = np.where(dataset['total_UPDRS']>=30, 1, 0)


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
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



