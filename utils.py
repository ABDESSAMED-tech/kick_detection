import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tslearn.metrics import dtw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,OneClassSVM
from sklearn.metrics import classification_report
def segmantation(X,y,window_length=36 ,step_size=1):
      # Define sliding window parameters
        # Amount of overlap between segments
      # Segment the time series data with sliding window
      segments = []
      labels = []

      for i in range(0, len(X) - window_length, step_size):
          segment = X[i:i+window_length]
          segments.append(segment)
          
          # Assign label to the segment based on the presence of anomalies
          segment_labels = y[i:i+window_length]
          if np.any(segment_labels == 1):
              label = 1  # Anomaly present
          else:
              label = 0  # No anomaly
          labels.append(label)

      # Convert segments and labels to numpy arrays
      segments = np.array(segments)
      labels = np.array(labels)
      return segments,labels
def split_data_balanced(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets with balanced labels.

    Parameters:
    - X: Input features (numpy array or pandas DataFrame)
    - y: Target labels (numpy array or pandas Series)
    - test_size: Proportion of the data to be used for testing (default: 0.2)
    - random_state: Random seed for reproducibility (default: None)

    Returns:
    - X_train: Training set features
    - X_test: Testing set features
    - y_train: Training set labels
    - y_test: Testing set labels
    """
    # Find unique labels and their counts
    unique_labels, label_counts = np.unique(y, return_counts=True)

    # Find the minimum label count
    min_label_count = np.min(label_counts)

    # Split the data for each label, ensuring balanced classes in the test set
    X_train, X_test, y_train, y_test = [], [], [], []
    for label in unique_labels:
        # Split the data for the current label
        X_label = X[y == label]
        y_label = y[y == label]
        X_label_train, X_label_test, y_label_train, y_label_test = train_test_split(
            X_label, y_label, test_size=test_size, random_state=random_state
        )

        # Add the split data to the overall train and test sets
        X_train.append(X_label_train)
        X_test.append(X_label_test)
        y_train.append(y_label_train)
        y_test.append(y_label_test)

    # Concatenate the data from all labels
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)

    return X_train, X_test, y_train, y_test

def Knn_algorithme(df,k,Features,target,test_size,window_size=36):
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments,labels = segmantation(X,y,window_length=window_size )
    print('tetetetettqjzkfqsdvjnkqsdvjkln',test_size/100)
    X_train, X_test, y_train, y_test = split_data_balanced(segments,labels, test_size=test_size/100 )
    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)
    # Reshape the feature matrices for Knn 
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model = KNeighborsClassifier(n_neighbors=k,metric=dtw)
    # Train the model
    model.fit(X_train, y_train)
        # Make predictions
    y_pred = model.predict(X_test)
    
    return classification_report(y_test, y_pred),y_test, y_pred
    

def LSTM_algorithm(df,epoch,Features,target,test_size,window_size):
    
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments,labels = segmantation(X,y,window_length=window_size )
    X_train, X_test, y_train, y_test = split_data_balanced(segments, labels, test_size=test_size/100)

    X_test , y_test = np.array(X_test), np.array(y_test).reshape(- 1 , 1 )
    X_train , y_train = np.array(X_train), np.array(y_train).reshape(-1,1 )
    data_dim = 4
    timesteps = 36
    num_classes = 2
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(8, return_sequences=True,
                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=1, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(loss, accuracy )
    return loss ,accuracy 

def SVM_algorithme(df,Features,target,c,Kernel,Gamma,test_size,window_size):
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments,labels = segmantation(X,y,window_length=window_size )
    print('tetetetettqjzkfqsdvjnkqsdvjkln',test_size/100)
    X_train, X_test, y_train, y_test = split_data_balanced(segments,labels, test_size=test_size/100 )
    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)
    # Reshape the feature matrices for Knn 
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(X_train.shape,X_test.shape)
    svm = OneClassSVM(kernel=Kernel, nu=c, gamma=Gamma)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return classification_report(y_test, y_pred)
    
    
    