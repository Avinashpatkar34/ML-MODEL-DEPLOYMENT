# import lib
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

#load dataset file dp variable_name
dp =pd.read_csv("iris.csv")
#print first 5 rows
print(dp.head())


#select dependent and independent variable
x=dp[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y=dp["Class"]

#split data into train test
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=50)

#feature scaling 
sc= StandardScaler()
X_train =sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Instantiate the model
cs =RandomForestClassifier()

#fit the model
cs.fit(X_train,y_train)


#pickle file 
pickle.dump(cs,open("pickle.pkl","wb"))

