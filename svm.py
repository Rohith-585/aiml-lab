from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data # Features: sepal length, sepal width, petal length, petal width
y = iris.target # Labels: three species of Iris
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the SVM classifier
svm_model = SVC(kernel='linear') # You can also try 'rbf', 'poly', etc.
# Train the SVM model on the training data
svm_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = svm_model.predict(X_test)
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred)) 