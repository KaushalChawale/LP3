# Implement K-Nearest Neighbors algorithm on diabetes.csv dataset. Compute confusion matrix, accuracy, error rate, 
# precision and recall on the given dataset. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

df = pd.read_csv("C:\\Users\Lenovo\Desktop\Study\Engineering\BE SEM 1\LP 3\ML\diabetes.csv")
print(df.head())

print('\n----------------------------------------\n')

print(df.columns)

print('\n----------------------------------------\n')

print(df.dtypes)

print('\n----------------------------------------\n')

#Splitting the data into features and target variable
X = df.drop('Outcome', axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Confusion matrix: \n', confusion_matrix(y_test,y_pred))

print('\n----------------------------------------\n')

print('Classification report:\n ' ,classification_report(y_test,y_pred))

print('\n----------------------------------------\n')

print('Accuracy score: ', accuracy_score(y_test,y_pred))

print('\n----------------------------------------\n')

sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, 
          xticklabels = ['No Diabetes','Diabetes'],
          yticklabels = ['No Diabetes','Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()