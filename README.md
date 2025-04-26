# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, remove irrelevant columns, and handle missing values and duplicates.
2. Encode categorical features using LabelEncoder.
3. Split the data into training and testing sets (80% train, 20% test).
4. Train a Logistic Regression model, predict on the test set, and calculate the accuracy.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Venkatachalam S
RegisterNumber:  212224220121

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/Placement_Data.csv')
df.head()

data = df.copy()
data = data.drop(['sl_no','salary'],axis = 1)
data.head()

data.isnull().sum()

data.duplicated().sum()

le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['ssc_b'] = le.fit_transform(data['ssc_b'])
data['hsc_b'] = le.fit_transform(data['hsc_b'])
data['hsc_s'] = le.fit_transform(data['hsc_s'])
data['degree_t'] = le.fit_transform(data['degree_t'])
data['workex'] = le.fit_transform(data['workex'])
data['specialisation'] = le.fit_transform(data['specialisation'])
data['status'] = le.fit_transform(data['status'])
data

x = data.iloc[:,:-1]
x

y = data['status']
y

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver = 'liblinear')
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_pred

accuracy = accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/user-attachments/assets/7e86731d-857d-41c6-83f2-564cbfe4715d)
![image](https://github.com/user-attachments/assets/2276bc30-cf09-4c51-8ef6-4b1e35b0d315)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
