import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle  #used to create pickle files
# import joblib  #used to save models
# from keras.layers import Input
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
# from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('heart_statlog_cleveland_hungary_final.csv');

X = dataset.drop('target', axis=1)
y = dataset['target']

print(dataset.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base estimators
base_estimators = [
    ('rf', RandomForestClassifier()),
    ('svc', SVC())
]

# Define the meta-classifier
meta_classifier = LogisticRegression()

# Create the StackingClassifier
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_classifier
)

stacking_clf.fit(X_train, y_train)

print(X_train)

# Make predictions
y_pred = stacking_clf.predict(X_test)

# Generate and print the classification report
report = classification_report(y_test, y_pred)
print(report)

print("creating pickle file...")
pickle.dump(stacking_clf,open('heart_disease_classifier.pkl','wb'))
print("Pickle file created")