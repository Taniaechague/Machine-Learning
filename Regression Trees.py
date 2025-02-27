### Regression Trees

## Import Libraries
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

## Dataset Analysis
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data
#To understand the dataset a little better, let us plot the correlation of the target variable against the input variables.
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

## Dataset Preprocessing
# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')
# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)
# get the feature matrix used for training
X = proc_data.values
# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

## Dataset Train/Test Split
#Now that the dataset is ready for building the classification models, you need to first divide the pre-processed dataset into a subset to be used for training the model (the train set) and a subset to be used for evaluating the quality of the model (the test set).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Build a Decision Tree Regressor model with Scikit-Learn
#Regression Trees are implemented using DecisionTreeRegressor. The important parameters of the model are:criterion: The function used to measure error, we use 'squared_error'. max_depth - The maximum depth the tree is allowed to take; we use 8.
# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor
# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8, 
                               random_state=35)
#Now lets train our model using the fit method on the DecisionTreeRegressor object providing our training data
dt_reg.fit(X_train, y_train)

## Evaluate the Scikit-Learn and Snap ML Decision Tree Regressor Models
#To evaluate our dataset we will use the score method of the DecisionTreeRegressor object providing our testing data, this number is the r-squared value which indicates the coefficient of determination. We will also evaluate the Mean Squared Error (MSE) of the regression output with respect to the test set target values. High r-squared and low MSE values are expected from a good regression model.
# run inference using the sklearn model
y_pred = dt_reg.predict(X_test)
# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))
r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

## What if we change the max_depth to 12? How would the and be affected?
#MSE is noted to be increased by increasing the max_depth of the tree. This may be because of the model having excessive parameters due to which it overfits to the training data, making the performance on the testing data poorer. Another important observation would be that the model gives a negative value of r-squared. This again indicates that the prediction model created does a very poor job of predicting the values on a test set.

## Identify the top 3 features with the most effect on the tip_amount.
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
abs(correlation_values).sort_values(ascending=False)[:3]
#As is evident from the output, Fare amount, toll amount and trip distance are the top features affecting the tip amount, which make logical sense.

## Since we identified 4 features which are not correlated with the target variable, try removing these variables from the input set and see the effect on the MSE and r-squared value.
raw_data = raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
# Execute all the cells of the lab after modifying the raw data.
#The MSE and r-squared values does not change significantly, showing that there is minimal affect of these parameters on the final regression output.

## Check the effect of decreasing the max_depth parameter to 4 on the MSE and r-squared values.
#You will note that the MSE value decreases and r-squared value increases, meaning that the choice of `max_depth=4` may be more suited for this dataset.


