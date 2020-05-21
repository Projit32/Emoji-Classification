# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import ShuffleSplit

"""pickle files"""

path="TweetDataset/"
with open(path+"TwitterFinalDataset.pickle", 'rb') as data:
    dataset = pickle.load(data)
with open(path+"X_test.pickle", 'rb') as data:
    X_test = pickle.load(data)
with open(path+"X_train.pickle", 'rb') as data:
    X_train = pickle.load(data)
with open(path+"y_test.pickle", 'rb') as data:
    y_test = pickle.load(data)
with open(path+"y_train.pickle", 'rb') as data:
    y_train = pickle.load(data)

import tensorflow
from tensorflow import constant_initializer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


# Initialising the ANN
ANN_classifier = Sequential()

#Adding Layers
#Hidden Layers
#1
ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform',input_shape=(750,)))
ANN_classifier.add(PReLU(alpha_initializer=constant_initializer(value=0.1)))
#2
ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform'))
ANN_classifier.add(PReLU(alpha_initializer=constant_initializer(value=0.1)))
#Output Layers (softmax for multi class prediction)
ANN_classifier.add(Dense(units=3, kernel_initializer  = 'uniform', activation = 'softmax'))
#Compile ANN
ANN_classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])


#Train
histroty=ANN_classifier.fit(X_train, y_train, batch_size = 500, epochs = 30, use_multiprocessing=True, validation_data=(X_test,y_test))

#Loading previous model (if any)
ANN_classifier=load_model(path+'ANNTFV3.1.h5')

acc=ANN_classifier.evaluate(X_test, y_test)

#Train-Test Accuracy plot
plt.plot(histroty.history['sparse_categorical_accuracy'], label='Training')
plt.plot(histroty.history['val_sparse_categorical_accuracy'], label='Testing')
plt.legend()
plt.show()


#Predictions
y_pred=ANN_classifier.predict_classes(X_test)

from sklearn.metrics import precision_recall_fscore_support

precision,recall,fscore,support=precision_recall_fscore_support(y_test,y_pred,labels=[0,1,2])
print('Labels: \tSexual      Offensive\tNeutral')
print('Precision:\t'+str(precision))
print('Recall:\t\t'+str(recall))
print('F1 Score:\t'+str(fscore))
print('Support:\t'+str(support))

#confusion matrix
aux_df = ['Sexual','Offensive','Neutral']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_matrix, 
            annot=True,
            xticklabels=aux_df, 
            yticklabels=aux_df,
            )
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()


#Evaluating and checking for possible overfiting
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
  ANN_classifier = Sequential()
  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform',input_shape=(750,)))
  ANN_classifier.add(PReLU(alpha_initializer=constant_initializer(value=0.1)))
  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform'))
  ANN_classifier.add(PReLU(alpha_initializer=constant_initializer(value=0.1)))
#  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform', activation = 'relu'))
#  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform', activation = 'relu'))
  ANN_classifier.add(Dense(units=1, kernel_initializer  = 'uniform', activation = 'softmax'))
  ANN_classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
  return ANN_classifier

cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42)
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 250, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = cv_sets, n_jobs = -1, verbose=100)


mean = accuracies.mean()
variance = accuracies.std()

"""
#Optimising
#Turning
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
  ANN_classifier = Sequential()
  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'glorot_uniform', activation = 'relu',input_shape=(1000,)))
  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform', activation = 'relu'))
#  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform', activation = 'relu'))
#  ANN_classifier.add(Dense(units=376, kernel_initializer  = 'uniform', activation = 'relu'))
  ANN_classifier.add(Dense(units=1, kernel_initializer  = 'uniform', activation = 'softmax'))
  ANN_classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
  return ANN_classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [250,200],
              'epochs': [100, 200],
              'optimizer': ['adam', 'rmsprop']}


cv_sets = ShuffleSplit(n_splits = 3, test_size = 0.2, random_state = 42)
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv =cv_sets)


grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

ANN_classifier= grid_search.best_estimator_

#Optimised Predictions
y_pred=ANN_classifier.predict(X_test)
y_pred = (y_pred > 0.65)


#confusion matrix
aux_df = dataset[['Category']].drop_duplicates().sort_values('Category')
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_matrix, 
            annot=True,
            xticklabels=aux_df['Category'].values, 
            yticklabels=aux_df['Category'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()


"""

#Final testing
X_train_pred=ANN_classifier.predict_classes(X_train)

#saving
d = {
     'Model': 'ANN TF 3 layer',
     'Training Set Accuracy': accuracy_score(y_train,X_train_pred),
     'Test Set Accuracy': accuracy_score(y_test, y_pred)
}

AccuracySet = pd.DataFrame(d, index=[0])



ANN_classifier.save(path+'ANNTFV3.1.h5')

with open(path+'ANNTFAccuracy.pickle', 'wb') as output:
    pickle.dump(AccuracySet, output)
