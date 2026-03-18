# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data Loading and Exploration
3. Data Preparation
4. Train-Test Split
5. Text Vectorization
6. Model Training
7. Making Predictions
8. Evaluation
9. End
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: TAMILPAGALAVAN K
RegisterNumber: 212223040224 
*/
import pandas as pd
data = pd.read_csv("spam.csv",encoding = 'Windows-1252') 
data.head()
data.isnull().sum()
x = data["v2"].values
y = data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:

<img width="748" height="219" alt="Screenshot 2026-03-16 082725" src="https://github.com/user-attachments/assets/27b540aa-017a-4342-ba44-f0198ef9af31" />

<img width="437" height="268" alt="Screenshot 2026-03-16 081815" src="https://github.com/user-attachments/assets/37778f73-5e10-45e8-80d9-753504727156" />

<img width="329" height="128" alt="Screenshot 2026-03-16 092809" src="https://github.com/user-attachments/assets/4b5fcfdb-29bf-4bcc-b3b6-1ba7ac6562cd" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
