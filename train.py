import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Importing Performance Metrics:
from sklearn.ensemble import RandomForestRegressor
import pickle


train_data = pd.read_csv('Train_Data.csv')

# Rounding up & down Age:
train_data['age'] = round(train_data['age'])

# Encoding:
train_data = pd.get_dummies(train_data, drop_first=True)

# Rearranging columns to see better: 
train_data = train_data[['age','sex_male','smoker_yes','bmi','children','region_northwest','region_southeast','region_southwest','charges']]
train_data.head(2)

# Splitting Independent & Dependent Feature:
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Random Forest Regressor:
RandomForestRegressor = RandomForestRegressor()
RandomForestRegressor = RandomForestRegressor.fit(X_train, y_train)

# Prediction:
y_pred = RandomForestRegressor.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# # Creating a pickle file for the classifier
filename = 'MedicalInsuranceCost.pkl'
pickle.dump(RandomForestRegressor, open(filename, 'wb'))