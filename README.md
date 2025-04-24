# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SABARINATH R
RegisterNumber: 212223100048
*/

import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()
data.info()
data.isnull().sum()
data['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=data['left']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
## Dataset
![image](https://github.com/user-attachments/assets/9fdaeb50-3dea-4649-b804-19df5366ae65)

## Information
![image](https://github.com/user-attachments/assets/537054aa-df68-47fd-93f6-87a88019a624)


## Non Null values
![image](https://github.com/user-attachments/assets/3bdae5d3-0528-482e-9851-519ad4bfbd65)


## Encoded value
![image](https://github.com/user-attachments/assets/81955841-5112-4da3-8b00-5584be1cece0)

## Count
![image](https://github.com/user-attachments/assets/095bbc56-162a-4105-809c-b61b144fa86a)


## X and Y value
![image](https://github.com/user-attachments/assets/34cbe86f-af67-469d-abed-de7cd7fc4fa4)

![image](https://github.com/user-attachments/assets/8ba85c65-3b35-4c3e-8a08-6fb595a5c7fe)

## Accuracy
![image](https://github.com/user-attachments/assets/4d719720-7250-4aba-8810-95e0fef75628)

## Predicted
![image](https://github.com/user-attachments/assets/9cc75c1d-5d31-4a8c-b0e7-4f181fb2bb10)

## Plot
![image](https://github.com/user-attachments/assets/ad41bb97-8270-4a26-a797-42c800af74b3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
