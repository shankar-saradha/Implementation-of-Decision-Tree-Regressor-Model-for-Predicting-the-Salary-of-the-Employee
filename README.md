# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```Python 
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Shankar S S 
RegisterNumber: 212221240052  
*/
import pandas as pd
d=pd.read_csv("Salary.csv")
d.head()
d.info()
d.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
d["Position"] = l.fit_transform(d["Position"])
d.head()

x = d[["Position","Level"]]
y = d["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
Head:

![172994812-0f716708-5a1a-4d90-8ab6-6f3533abfb34](https://user-images.githubusercontent.com/93978702/173190617-44661f46-ae9a-4672-a2b8-58ec1b67acfc.jpg)

Info:

![172994827-302d0bde-6f49-4573-99e0-9cfa2c592ce3](https://user-images.githubusercontent.com/93978702/173190624-693a277f-977d-4300-8331-db1ac0cdd6cc.jpg)

Head using label encoder:

![172994945-f9c437cb-13ed-4016-b618-504c685e7725](https://user-images.githubusercontent.com/93978702/173190679-bb3d3aac-7a49-4f65-9b6a-e7ad5a5b598b.jpg)

Mean square error:

![172994945-f9c437cb-13ed-4016-b618-504c685e7725](https://user-images.githubusercontent.com/93978702/173190718-db1679bc-e8d2-4a77-b9c2-250345d659ed.jpg)


r2:

![172995188-a43e367b-08af-4841-a5fa-a787c7a2936e](https://user-images.githubusercontent.com/93978702/173190691-50aba3c7-59ad-40d6-a73a-ee9264532143.jpg)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
