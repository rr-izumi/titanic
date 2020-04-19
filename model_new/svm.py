'''remove columns of PassengerId,Cabin,TicketID,Name'''

import pandas as pd
import numpy as np
import numpy.random as rondom
import scipy as sp
from sklearn.model_selection import train_test_split

train = pd.read_csv('../data/input/train.csv')
test_pre = pd.read_csv('../data/input/test.csv')

train.Sex = train.Sex.replace('male',1).replace('female',2)
train.Embarked = train.Embarked.replace('S',1).replace('C',2).replace('Q',3)
train.Embarked = train.Embarked.fillna('1')
train.Age = train.Age.fillna(train.Age.median())
test_pre.Sex = test_pre.Sex.replace('male',1).replace('female',2)
test_pre.Embarked = test_pre.Embarked.replace('S',1).replace('C',2).replace('Q',3)
test_pre.Fare = test_pre.Fare.fillna(test_pre.Fare.median())
test_pre.Age = test_pre.Age.fillna(test_pre.Age.median())

train_Y = train['Survived']
X = train.drop("Survived", axis = 1).drop('PassengerId', axis=1).drop('Name', axis=1).drop('Cabin', axis=1).drop('Ticket', axis=1)
test = test_pre.drop('PassengerId', axis=1).drop("Name", axis=1).drop("Cabin", axis=1).drop('Ticket', axis=1)

from sklearn import preprocessing

#標準化のインスタンス作成
ss = preprocessing.StandardScaler()#分散は標準分散しか適用できない
train_X = pd.DataFrame(ss.fit_transform(X), index=X.index, columns=X.columns)#fit_transformはndarrayに変換されるためdfに入れ直し
test_X = pd.DataFrame(ss.fit_transform(test), index=test.index, columns=test.columns)

print('normalized train_X\n',train_X.head())
print('normalized test_X\n',test_X.head())

print('train_X_shape:',train_X.shape)
print('test_X_shape:',test_X.shape)

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV

search_params = [{
'C' : np.linspace(1,10,80),
'kernel' : ['linear', 'sigmoid', 'rbf'], #poly
#degree : np.arange(1,5,1) # for poly
'gamma' : np.linspace(0.01, 0.1, 50),
'random_state' : [1] ,
}]

#model = LinearSVC()
model = SVC()
gs = GridSearchCV(model, search_params, cv=3, verbose=True,)
gs.fit(train_X, train_Y)
print('best_score:',gs.best_score_)
print('best estimater:', gs.best_estimator_)
test_Y = gs.best_estimator_.predict(test_X)
submission = pd.DataFrame({'PassengerId': test_pre['PassengerId'], 'Survived': test_Y})
submission.to_csv('../data/output/SVM.csv', index=False)
