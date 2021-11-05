"""Import the libraries"""
import pandas as pd
from matplotlib import pyplot as ply
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('Prostate_Cancer.csv')
data.drop(['id','fractal_dimension', 'texture', 'perimeter'], inplace=True, axis=1)

#肿瘤半径， 肿瘤面积， 表面平滑度， 密实度， 对称度

#Our target will be "diagnosis_result"
data['diagnosis_result'].replace({'M':'Maligne','B':'Benigne'},inplace=True)

#train test split
X = data.drop(['diagnosis_result'], axis=1) # Features
y = data['diagnosis_result'] # Labels
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)
"""
forest = RandomForestClassifier(n_estimators = 50)
forest.fit(X,y)
"""
y_pred= forest.predict(X_test)
#See accuracy
#print(accuracy_score(y_test, y_pred))
"""

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(forest, open('model.pkl','wb'))

"""
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[23,954,0.14,0.278,0.242]]))
"""
