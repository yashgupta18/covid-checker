import pandas as pd
import numpy as np
import random
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv(r"/Users/yashgupta/Desktop/covid-19/COVID_data.csv")
data.shape
data.columns
data.head()
features=data.columns
x=data.iloc[:,:-1]
x
y=data.iloc[:,-1:]
y

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

covid=DecisionTreeClassifier(criterion="entropy",splitter="random",random_state=101)
X_test
covid.fit(X_train,y_train)
y_predicted=covid.predict(X_test)
y_predicted[:10]
print(accuracy_score(y_test,y_predicted))
print(f1_score(y_test, y_predicted,average='macro'))
print(f1_score(y_test, y_predicted,average='weighted'))
print(f1_score(y_test, y_predicted,average='micro'))
X_test1=[]
covid.predict([[38,0,101,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1]])


# Pkl_Filename = 'finalized_model.pkl'
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(covid, file)

# inputt=[int(x) for x in "45 32 60".split(' ')]
# final=[np.array(inputt)]

# b = log_reg.predict_proba(final)

pickle.dump(covid,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))









