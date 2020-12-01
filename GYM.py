# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:52:54 2020

@author: Ali
"""

import seaborn as sns
import pandas as pd
import numpy as np
import sklearn 

df = pd.read_excel(r'C:/Users/Ali/Downloads/dataGYM.xlsx')
df.head(3)

df['Class'].replace(to_replace='EXtremely obese', value='Extremely obese', inplace=True)
print(df['Class'].value_counts())

data = df.copy(deep=True)
del data['Class']
del data['BMI']
print(data.head(3))

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
print(X)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.1 , random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


s = y.unique()
s

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(oob_score=True , bootstrap=True,
                            n_jobs=-1 , n_estimators = 400, max_features=2 , criterion='entropy', max_depth=12 ,
                             class_weight={s[1]:2.845 , s[3]: 2.645 ,
                                           s[2]:0.734, s[0]:0.234, s[4]: 1.234},
                             
                             random_state=42)
clf.fit(X_train,y_train)

predd = clf.predict(X_test)
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(clf.score(X_train, y_train), 
                                                                                             clf.oob_score_,
                                                                                             clf.score(X_test, y_test)))
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test , predd)
print(acc)


from sklearn import metrics
print(metrics.classification_report(y_test, predd))
metrics.confusion_matrix(y_test, predd)

from sklearn.model_selection import cross_val_score 
accuracies = cross_val_score(estimator = clf, X = X_train,\
     y = y_train, cv = 15)
print("Accuracy Mean {} Accuracy Variance \
     {}".format(accuracies.mean(),accuracies.std()))
accuracies



# We don't use it ...
import pickle

pickle.dump(clf, open("Ali-gym-diet.pkl", "wb"))

model = pickle.load(open("Ali-gym-diet.pkl", "rb"))

print(model.predict([[40,5.6,70]]))










