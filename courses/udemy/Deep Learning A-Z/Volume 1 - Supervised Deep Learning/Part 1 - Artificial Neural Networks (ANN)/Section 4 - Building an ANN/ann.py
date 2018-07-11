import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Part 1 - Data Preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

label_encoder = LabelEncoder()
x[:, 1] = label_encoder.fit_transform(x[:, 1])
x[:, 2] = label_encoder.fit_transform(x[:, 2])

onehot_encoder = OneHotEncoder(categorical_features = [1])
x = onehot_encoder.fit_transform(x).toarray()
x = x[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# Part 2 - Building the ANN
clf = Sequential()
clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))
clf.add(Dropout(rate=0.1))
clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
clf.add(Dropout(rate=0.1))
clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

clf.fit(x_train, y_train, batch_size=10, epochs=20)

# Part 3 - Predictions and evaluation of the model
y_pred = clf.predict(x_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Homework answer
customer = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
customer[:, 1] = label_encoder.fit_transform(customer[:, 1])
customer[:, 2] = label_encoder.fit_transform(customer[:, 2])
customer = onehot_encoder.transform(customer).toarray()
customer = std.transform(customer)

print("Should we say goodbye to the customer?", clf.predict(customer) > 0.5)

# Part 4 - Evaluating, Improving and Tuning the ANN
def build_classifier(optimizer):
    clf = Sequential()
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))
    clf.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    clf.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return clf

clf_keras = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=20)
accuracies = cross_val_score(estimator=clf_keras, X=x_train, y=y_train, cv=10, verbose=10, n_jobs=-1)
print(accuracies.mean(), accuracies.std())

clf_keras = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32], 
              'epochs': [10, 20],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=clf_keras, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_, grid_search.best_score_)