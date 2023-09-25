# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MOHAMED ATHIL B
RegisterNumber: 212222230081
*/
```
```

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### placement data:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/72297679-c5e4-451f-a9e4-7fe194dad9f9)
### salary data:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/80b70958-640c-435b-aa80-09ba31afc0b4)
### checking null function:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/4efc61b2-8cfa-431d-b100-b2151be1bcb0)
### Data Duplicate:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/9e2ab69f-27ca-4ff7-b9a4-306f18ef66da)
### print data:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/ca64a9ab-a9e7-4216-b712-a8080cca326e)
### Data Status:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/8a5fce13-2b63-4995-9c01-e853406a326f)
### y_prediction array:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/6b7906e7-2267-4edf-9f3d-352ab0576332)
### Accuracy value:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/c0d61462-0856-4eba-8e79-b623bdd8694a)
### confusion matrix:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/c17a2e68-d4f0-4d63-86c9-0769f274ac73)
### classification report:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/c9d1a442-d90f-4a40-85b2-36f60a46d8d9)
### prediction of LR:
![image](https://github.com/Bmohamedathil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119560261/a46416fe-16e5-47f0-b83e-fa0fd400ffab)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
