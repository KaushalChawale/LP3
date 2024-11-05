# Implementing Linear Regression and Random Forest Regressor methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

#import the dataset
df = pd.read_csv(r"C:\\Users\Lenovo\Desktop\Study\Engineering\BE SEM 1\LP 3\ML\uber.csv")
print(df)
print('\n----------------------------------------\n')

#Showing the data types
print(df.dtypes)
print('\n----------------------------------------\n')


#changing the data types
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
print(df.dtypes)
print('\n----------------------------------------\n')

#displaying null values
print(df.isna().sum())
print('\n----------------------------------------\n')


# dropping rows with null values
df.dropna(inplace=True)
print("After handling null values: \n",df.isna().sum())
print('\n----------------------------------------\n')

#Duplicate values
print('Duplicate values:',df.duplicated().sum())
print('\n----------------------------------------\n')

#Outliers visualization
sns.boxplot(x=df['fare_amount'])
plt.title('Fare_amount with outliers')
plt.show()

#Removing outliers
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - IQR * 1.5
upper_bound = Q3 + IQR * 1.5

df_no_outliers = df[(df['fare_amount'] >= lower_bound) & (df['fare_amount'] <= upper_bound)]

#Fare_amount visualization after removing outliers
sns.boxplot(x = df_no_outliers['fare_amount'])
plt.title('Fare_amount without outliers')
plt.show()

#Checking the correlation
correlation_matrix = df_no_outliers.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Splitting the data into training and testing sets
X = df_no_outliers[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y = df_no_outliers['fare_amount']

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Implementing Linear model and Random Forest model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train,y_train)

#Evaluating the model
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

#Calculating r2_score and RMSE for both models

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = root_mean_squared_error(y_test,y_pred_lr)

r2_rf = r2_score(y_test,y_pred_rf)
rmse_rf = root_mean_squared_error(y_test,y_pred_rf)

#Printing the results
print('Linear regression r2: ', r2_lr)
print('Linear Regression RMSE: ', rmse_lr)

print('Random forest regression r2: ', r2_rf)
print('Random forest RMSE: ', rmse_rf)
