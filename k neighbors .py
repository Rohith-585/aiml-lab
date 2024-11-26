from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
print(X)

y = pd.DataFrame(iris.target)
y.columns = ['Targets']
print(y)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)
print("Dataset is split into training and testing...")
print("Size of training data and its label", x_train.shape, y_train.shape)
print("Size of testing data and its label", x_test.shape, y_test.shape)

for i in range(len(iris.target_names)):
    print("Label", i, "-", str(iris.target_names[i]))

classifer = KNeighborsClassifier(n_neighbors=3)
classifer.fit(x_train, y_train)

y_pred = classifer.predict(x_test)

print("Results of Classification using K-nn with K=3")
for r in range(0, len(x_test)):
    print(" sample:", str(x_test[r]), " Actual-label:", str(y_test[r]), " predict-label:", str(y_pred[r]))

print("Classification Accuracy:", classifer.score(x_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Accuracy Metrics')
print(classification_report(y_test, y_pred))
