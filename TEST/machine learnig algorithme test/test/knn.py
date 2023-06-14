import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit

df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\data\Well-26_06-09-2020.xlsx')

X = df[['TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']].values
y = df['STATUS'].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Define time steps and target horizon
window_size =36  # Number of time steps to consider (5 seconds * 36 = 3 minutes)
target_horizon =1 # Number of time steps ahead to predict (5 seconds * 6 = 30 seconds)

# Generate input sequences and corresponding targets
# X_sequences = []
# y_targets = []
# for i in range(len(X_scaled) - window_size - target_horizon + 1):
#     X_sequences.append(X_scaled[i:i+window_size])
#     y_targets.append(y[i+window_size+target_horizon-1])

# X_sequences = np.array(X_sequences)
# y_targets = np.array(y_targets)
# Reshape the input data for oversampling
print(X_sequences.shape)
print(y_targets.shape)

n_samples, n_timesteps, n_features = X_sequences.shape
X_reshaped = X_sequences.reshape(n_samples, n_timesteps * n_features)
n_splits=10
skf = StratifiedKFold(n_splits)
# Initialize lists to store evaluation metrics
confusion_matrices = []
precisions = []
recalls = []
f1_scores = []
accuracies = []
tscv = TimeSeriesSplit(n_splits=n_splits)
# Iterate through each split
for train_index, test_index in tscv.split(X_reshaped,y_targets):
    # Split the data into training and testing sets
    X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
    y_train, y_test = y_targets[train_index], y_targets[test_index]
    print(set(y_train),set(y_test))
    # unique_elements, counts = np.unique(y_test, return_counts=True)
    # train,count= np.unique(y_train, return_counts=True)
    # # Print the frequencies
    # for element, count in zip(unique_elements, counts):
    #     print(f"y_test: {element}: {count}")
    # for element, count in zip(train,count):
    #     print(f"y_train{element}: {count}")
    # Build the k-NN classifier
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

    # Append the metrics to the respective lists
    confusion_matrices.append(cm)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    accuracies.append(accuracy)

aggregate_cm = sum(confusion_matrices)

print('Confusion Matrix:\n', aggregate_cm)
print('Precision:', sum(precisions)/n_splits)
print('Recall:', sum(recalls)/10)
print('F1-Score:', sum(f1_scores)/n_splits)
print('Accuracy:', sum(accuracies)/n_splits)
print('-----------------------')

print('Confusion Matrix:\n', aggregate_cm)
print('Precision:', (precisions))
print('Recall:', (recalls))
print('F1-Score:', (f1_scores))
print('Accuracy:', (accuracies))

print('-----------------------')