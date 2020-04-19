'''remove columns of PassengerId,Cabin,TicketID,Name'''

import pandas as pd
import numpy as np
import numpy.random as rondom
import scipy as sp
from sklearn.model_selection import train_test_split

train = pd.read_csv('../data/input/train.csv')
test = pd.read_csv('../data/input/test.csv')

train.Sex = train.Sex.replace('male',1).replace('female',2)
train.Embarked = train.Embarked.replace('S',1).replace('C',2).replace('Q',3)
train.Embarked = train.Embarked.fillna('1')
train.Age = train.Age.fillna(train.Age.median())
test.Sex = test.Sex.replace('male',1).replace('female',2)
test.Embarked = test.Embarked.replace('S',1).replace('C',2).replace('Q',3)
test.Fare = test.Fare.fillna(test.Fare.median())
test.Age = test.Age.fillna(test.Age.median())

Y = train['Survived']
X = train.drop("Survived", axis = 1).drop('PassengerId', axis=1).drop('Name', axis=1).drop('Cabin', axis=1).drop('Ticket', axis=1)
test_X = test.drop('PassengerId', axis=1).drop("Name", axis=1).drop("Cabin", axis=1).drop('Ticket', axis=1)

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, random_state=0)

print('train_X_shape:',train_X.shape)
print('valid_X_shape:',valid_X.shape)
print('test_X_shape:',test_X.shape)

from sklearn.neighbors import KNeighborsClassifier

# 最適なkを探す
from sklearn import metrics
import matplotlib.pyplot as plt

accuracy =[]
for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X, train_Y)
    Y_pred = knn.predict(valid_X)
    accuracy.append(metrics.accuracy_score(valid_Y, Y_pred))
    print('K=',k,'accuracy_score:',metrics.accuracy_score(valid_Y, Y_pred))

#表示されない
#k_range = np.array(range(1,10))
#plt.plot(k_range, accuracy)
#plt.show()

knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(train_X, train_Y)

test_Y = knn.predict(test_X)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_Y})
submission.to_csv('../data/output/knn_200413.csv', index=False)
