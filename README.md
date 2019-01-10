# Orange Telecom Churned Or Not


## Imported all the required library
```
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
```
### Loading and Viewing the data

~~~
data=pd.read_csv('Orange_Telecom_Churn_Data.csv')
data.head()
~~~
# Data Visualisation

### Ploting the churned people

![alt Churned](https://github.com/rahuljadli/Orange-Telecom-Churned-Or-Not/blob/master/screen_shots/no_of_people.png)

### Ploting Area code Wise Churning

![alt Area code](https://github.com/rahuljadli/Orange-Telecom-Churned-Or-Not/blob/master/screen_shots/Area_code.png)

### Ploting Effect of intial plan

![alt initial plan ](https://github.com/rahuljadli/Orange-Telecom-Churned-Or-Not/blob/master/screen_shots/effect_of_intl_plan.png)

### Ploting Effect of voice plan

![alt Voice plan ](https://github.com/rahuljadli/Orange-Telecom-Churned-Or-Not/blob/master/screen_shots/effect_of_voice_plan.png)

### Ploting Effect of number of service calls

![alt Service calls ](https://github.com/rahuljadli/Orange-Telecom-Churned-Or-Not/blob/master/screen_shots/effect_of_no_of_service_call.png)


# Using Different Model's 

## Creating Training and Testing Data set

~~~
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

~~~
# Training the model using Logistic Regression

~~~
model=LogisticRegression()
model.fit(x_train,y_train)
~~~
# Making the prediction

~~~
new_prediction=model.predict(testing_data)
~~~
## Getting the accuracy score

~~~
from sklearn.metrics import accuracy_score


acc_logreg = round(accuracy_score(prediction, y_test) * 100, 2)
print(acc_logreg)

~~~
## got 86.29 using Logistic regression

# Training the model using KNN

~~~
model=KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train)
~~~
# Making the prediction

~~~
new_prediction=model.predict(testing_data)
~~~
## Getting the accuracy score

~~~
from sklearn.metrics import accuracy_score


acc_logreg = round(accuracy_score(prediction, y_test) * 100, 2)
print(acc_logreg)

~~~
## got 88.86 using KNN

# Training the model using Decision Tree Classifier

~~~
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
~~~
# Making the prediction

~~~
new_prediction=model.predict(testing_data)
~~~
## Getting the accuracy score

~~~
from sklearn.metrics import accuracy_score


acc_logreg = round(accuracy_score(prediction, y_test) * 100, 2)
print(acc_logreg)

~~~
## got 92.35 using Decision Tree Classifier

