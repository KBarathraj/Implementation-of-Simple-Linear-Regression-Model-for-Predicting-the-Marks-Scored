# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### **Algorithm: Predict Student Scores Using Linear Regression**

1. **Import Libraries**
   Import `pandas`, `numpy`, `matplotlib`, and required modules from `sklearn`.

2. **Load Dataset**
   Read `student_scores.csv` into a DataFrame.

3. **Prepare Data**

   * Define `X` = Hours studied
   * Define `y` = Scores

4. **Split Data**
   Split into training and test sets (80% train, 20% test).

5. **Train Model**
   Fit a `LinearRegression` model on training data.

6. **Make Predictions**
   Predict scores for test data.

7. **Visualize Results**

   * Plot training data and regression line
   * Plot test data and regression line

8. **Evaluate Model**
   Calculate and print:

   * MAE (Mean Absolute Error)
   * MSE (Mean Squared Error)
   * RMSE (Root Mean Squared Error)



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BARATHRAJ K
RegisterNumber:  212224230033
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv("student_scores.csv")
df.head()
X = df[['Hours']]
y = df['Scores']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Predicted Values: ")
print(y_pred)
print("Actual Values : ")
print(y_test)
plt.scatter(X_train,y_train,color="blue",label="training plot")
plt.plot(X_train,model.predict(X_train),color="red",label="Regression Line")
plt.title("Training Data:Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()
plt.scatter(X_test,y_test,color="blue",label="test plot")
plt.plot(X_test,y_pred,color="red",label="regression line")
plt.xlabel("hours")
plt.ylabel("scores")
plt.legend()
plt.show()
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```

## Output:
<img width="1473" height="1267" alt="image" src="https://github.com/user-attachments/assets/6b481ca2-8554-4351-9a72-9d1b95ba2119" />
<img width="1473" height="980" alt="image" src="https://github.com/user-attachments/assets/d5bd74ce-5400-4471-bf98-773fd1e679f1" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
