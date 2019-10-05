import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# import data and create numpy arrays
dataset = pd.read_csv("data/parkinsons_updrs.data",header=0)
X = dataset.iloc[:,dataset.columns != 'total_URDPS'].values
y = dataset.iloc[:,[5]].values


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


# Make PCA to reduce 
ct = ColumnTransformer(
    [('standard_scaler',StandardScaler(),[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])],    
    remainder='passthrough'                         
)
X = np.array(ct.fit_transform(X), dtype=np.float)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)


# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size = 0.2, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



for i in pca.explained_variance_ratio_:
    print('{0:.10f}'.format(i))
    
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
