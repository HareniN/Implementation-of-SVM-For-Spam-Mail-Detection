# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages. 

2.Import the dataset to operate on. 

3.Split the dataset.

4.Predict the required output.

5.End the program.
## Program:

/*
Program to implement the SVM For Spam Mail Detection.

Developed by: Hareni N

RegisterNumber: 212224040096 
*/
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

<img width="658" height="263" alt="image" src="https://github.com/user-attachments/assets/2fd6770d-5497-443b-a562-ba2dc457a03a" />
__________________________________________________________________________________________________________________________________

<img width="366" height="239" alt="image" src="https://github.com/user-attachments/assets/6757ef06-54b5-4015-90e2-22703de3079e" />
__________________________________________________________________________________________________________________________________

<img width="247" height="270" alt="image" src="https://github.com/user-attachments/assets/271b5339-a54a-4151-8dd9-237f50015203" />
__________________________________________________________________________________________________________________________________

<img width="696" height="329" alt="image" src="https://github.com/user-attachments/assets/29074ad6-cb2f-43ce-9bc5-da09c8421214" />
_________________________________________________________________________________________________________________________________


<img width="390" height="97" alt="image" src="https://github.com/user-attachments/assets/9c308ef3-1452-45db-b873-85fdb5debcb7" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
