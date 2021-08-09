import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
df=pd.read_csv('data_5.csv')
X = df.values[:, 1:4]
Y = df.values[:,0]
X_train, X_test, y_train, y_test= train_test_split( X, Y, test_size= 0.3, random_state= 100)
clf_entropy= DecisionTreeClassifier(criterion = "entropy", random_state= 100, 
max_depth=2, min_samples_leaf=3)
clf_entropy.fit(X_train, y_train)
y_pred_en= clf_entropy.predict(X_test)
print(y_pred_en)
print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
