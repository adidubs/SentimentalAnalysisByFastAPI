# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the data file
data_file = "feedbackS.csv"  # Adjust this according to your file path
df = pd.read_csv(data_file, encoding='latin1')  # Specify the encoding here

# Data preprocessing
X = df['comment'] # Assuming the column name is 'comment'
y = df['quality'] # Assuming the column name is 'quality'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vect, y_train)
nb_pred = nb_classifier.predict(X_test_vect)

# Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_vect, y_train)
svm_pred = svm_classifier.predict(X_test_vect)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_vect, y_train)
rf_pred = rf_classifier.predict(X_test_vect)

# Evaluate the models
print("Support Vector Machine (SVM) Classifier:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Classification Report:")
print(classification_report(y_test, svm_pred))

print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nNaive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Classification Report:")
print(classification_report(y_test, nb_pred))

