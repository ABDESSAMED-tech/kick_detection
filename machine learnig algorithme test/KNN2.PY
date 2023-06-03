import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split

df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\data\Well-26_06-09-2020.xlsx')

X = df[['TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']]
y = df['STATUS']
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)


train_features , test_features , train_labels , test_labels = train_test_split(X, y, test_size= 0.38, shuffle=False)
T = 45 # my choice of the rolling window
prepend_features = train_features.iloc[-(T- 1 ):]
test_features = pd.concat([prepend_features , test_features ], axis=0)
X_train , y_train = [], []
for i in range(train_labels.shape[ 0 ] - (T- 1 )):
    X_train.append(train_features.iloc[i:i+T].values)
    y_train.append(train_labels.iloc[i + (T- 1 )])
X_train , y_train = np.array(X_train), np.array(y_train).reshape(- 1 ,1 )
print(f'Train_data_dimensions:{X_train.shape},{y_train.shape}')
X_test , y_test = [], []
for i in range(test_labels.shape[ 0 ]):
    X_test.append(test_features.iloc[i:i+T].values)
    y_test.append(test_labels.iloc[i])
X_test , y_test = np.array(X_test), np.array(y_test).reshape(- 1 , 1 )
print(f'Test_data_dimensions:{X_test.shape},{y_test.shape}')

model = KNeighborsClassifier(n_neighbors=3)

    # Train the model
model.fit(X_train, y_train)

    # Make predictions
y_pred = model.predict(X_test)

    # Compute evaluation metrics
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

   

print('Confusion Matrix:\n', cm)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print('Accuracy:',accuracy)
print('-----------------------')