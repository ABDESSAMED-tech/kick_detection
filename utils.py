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
  
def Knn_algorithme(df,k,Features,target,test_size):
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments,labels = segmantation(X,y,window_length=12 )
    print('tetetetettqjzkfqsdvjnkqsdvjkln',test_size/100)
    X_train, X_test, y_train, y_test = train_test_split(segments,labels, test_size=test_size/100, shuffle=True)
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
    

def LSTM_algorithm(df,epoch,Features,target):
    
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments,labels = segmantation(X,y,window_length=36 )
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.3, shuffle=True)

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
    
    