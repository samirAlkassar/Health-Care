import scikitplot as skplt
from sklearn.metrics import confusion_matrix
import sklearn
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from sklearn import svm

data = pd.read_csv('car.data')
print(data.head())

data.info()
data.dtypes
data.describe()
data.shape
data.isna().sum()
data.isna().sum().sum()

data1 = data.fillna(method='pad')
data1
data1.isna().sum().sum()
data1['class'].value_counts()
data1['class'].value_counts().plot(kind='bar', figsize=(10, 5))
data2 = data1
data2.head()
le = preprocessing.LabelEncoder()
# buying=le.fit_transform(list(data1["buying"]))
# maint=le.fit_transform(list(data1["maint"]))
# door=le.fit_transform(list(data1["door"]))
# persons=le.fit_transform(list(data1["persons"]))
# lug_boot=le.fit_transform(list(data1["lug_boot"]))
# safty=le.fit_transform(list(data1["safty"]))
# cls=le.fit_transform(list(data1["class"]))
# predict='class'
for i in data2.columns:
    data2[i] = le.fit_transform(data2[i])

data2.hist()
plt.show()
plt.figure(figsize=(20, 20))

data2['buying']
data2.dtypes
data2.tail()
data2.describe()

# correlation measurement
corr_matrix = data2.corr()
corr_matrix["buying"].sort_values(ascending=False)

# features
x = data2[['buying', 'maint', 'door', 'persons', 'lug_boot', 'safty']]
# lables
y = data2[['class']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=5)
print(x_train)
print('================================================================================================')
print(y_train)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train.values, y_train.values.ravel())

# using svm
# classifier = svm.SVC(kernel='linear')
# classifier.fit(x_train, y_train)

# make a prediction for an example of an out-of-sample observation
output = knn.predict([[1, 1, 3, 2, 0, 0]])
if output == 1:
    print('good')
elif output == 3:
    print('verygoot')
elif output == 0:
    print('acc')
else:
    print("unacc")

x_train_prediction = knn.predict(x_train.values)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy score of the training data:', training_data_accuracy)

x_test_prediction = knn.predict(x_test.values)
testing_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of the Testing data:', testing_data_accuracy)

k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, np.ravel(y, order='C'))
    y_pred = knn.predict(x)
    scores.append(metrics.accuracy_score(y, y_pred))
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, np.ravel(y_train, order='C'))
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

confmatrix = confusion_matrix(y_test, x_test_prediction)
print(confmatrix)

skplt.metrics.plot_confusion_matrix(y_test, x_test_prediction, normalize=False)
plt.show()

classification = classification_report(y_test, x_test_prediction)
print("Classification Report \n", classification)

# make pickle file of qur model
pickle.dump(knn, open("model.pkl", "wb"))
