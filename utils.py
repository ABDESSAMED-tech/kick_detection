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
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import json
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
import pickle
# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def segmantation(X, y, window_length=36, step_size=1):
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
    return segments, labels


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


def Knn_algorithme(df, k, Features, target, test_size, window_size=36):
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments, labels = segmantation(X, y, window_length=window_size)
    print('tetetetettqjzkfqsdvjnkqsdvjkln', test_size/100)
    X_train, X_test, y_train, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    # Reshape the feature matrices for Knn
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model = KNeighborsClassifier(n_neighbors=k, metric=dtw
                                 )
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    print('im in prediction ...')
    y_pred = model.predict(X_test)
    print('end prediction')
    return y_test, y_pred


def segmentation_prediction(x, label, window, p):
    segments = []
    labels = []
    i = 0
    while i < len(x) - window + 1:
        segment = x[i:i+window]
        segments.append(segment)
        segment_labels = label[i:i+window]
        anomaly_percentage = np.mean(segment_labels)
        if 0 < anomaly_percentage <= p:
            # print(anomaly_percentage, 'if', i)
            labels.append(1)
        elif anomaly_percentage == 0:
            labels.append(0)
        else:
            labels.append(0)
            while True:
                i += 1
                if i >= len(x) - window or label[i] == 0:
                    break
        i += 1
        if i + window >= len(x):
            break
    # Convert segments and labels to numpy arrays
    segments = np.array(segments)
    labels = np.array(labels)
    return segments, labels


def LSTM_algorithm_detection(df, epoch, batch_size, Features, target, test_size, window_size):
    print('LSTM for DETECTION ... ')

    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments, labels = segmantation(X, y, window_length=window_size)
    X_train, X_test, y_train, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)

    X_test, y_test = np.array(X_test), np.array(y_test).reshape(- 1, 1)
    X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1, 1)
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(LSTM(units=64, input_shape=(
        window_size, 4), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), batch_size=batch_size, epochs=epoch, callbacks=[early_stopping])
    # val_loss, val_accuracy = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    y_pred = (predictions >= 0.5).astype(int)
    print('LSTM for DETECTION END ')
    return model, history, y_pred, y_test


def LSTM_algorithm_prediction(df, epoch, batch_size, Features, target, test_size, window_size):
    print('LSTM for PREDICTION .... ')

    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments, labels = segmentation_prediction(X, y, window_size, p=0.2)
    X_train, X_test, y_train, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)

    X_test, y_test = np.array(X_test), np.array(y_test).reshape(- 1, 1)
    X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1, 1)
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(LSTM(units=64, input_shape=(
        window_size, 4), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), batch_size=batch_size, epochs=epoch, callbacks=[early_stopping])
    # val_loss, val_accuracy = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    y_pred = (predictions >= 0.5).astype(int)
    print('LSTM for PREDICTION END  ')
    return model, history, y_pred, y_test


def SVM_algorithme_detection(df, Features, target, c, Kernel, Gamma, test_size, window_size):
    print('SVM for detection ... ')

    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments, labels = segmantation(X, y, window_length=window_size)
    X_train, X_test, y_train, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)
    # Reshape the feature matrices for Knn
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    svm = SVC(kernel=Kernel, C=c, gamma=Gamma)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print('SVM for detection end ')

    return svm,y_test, y_pred


def SVM_algorithme_prediction(df, Features, target, c, Kernel, Gamma, test_size, window_size):
    print('svm for prediction ... ')

    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments, labels = segmentation_prediction(X, y, window_size, p=0.2)
    X_train, X_test, y_train, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)

    # Reshape the feature matrices for Knn
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    svm = SVC(kernel=Kernel, C=c, gamma=Gamma)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print('SVM for prediction end ')

    return svm,y_test, y_pred


def RandomForest_detection(df, Features, target, test_size, window_size, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    print('Random forest for detection ... ')
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments, labels = segmantation(X, y, window_length=window_size)
    X_train, X_test, y_train, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf.predict(X_test)
    print('Random forest for detection end ')

    return rf, y_test, y_pred


def RandomForest_prediction(df, Features, target, test_size, window_size, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    print('Random forest for prediction .... ')
    X = df[Features].values
    y = df[target].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    segments, labels = segmentation_prediction(X, y, window_size, p=0.2)
    X_train, X_test, y_train, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf.predict(X_test)
    print('Random forest for detection end ')
    return rf,y_test, y_pred




# def prepare_data_svm_random_forest(df, Features, target, p,window,test_size):
#     save_path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\normalization_params_LSTM.json'
#     with open(save_path, 'r') as f:
#         normalization_params = json.load(f)
#     loaded_min = np.array(normalization_params['min'])
#     loaded_max = np.array(normalization_params['max'])
#     X = df[Features].values
#     y = df[target].values
#     X = (X - loaded_min) / (loaded_max - loaded_min)
#     if p == 'Detection':
#         segments, labels = segmantation(X, y, window_length=window)
#         X_train, X_test, y_train, y_test = split_data_balanced(segments, labels, test_size=test_size//100)
#         # Reshape the feature matrices for SVM
#         X_train = X_train.reshape(X_train.shape[0], -1)
#         X_test = X_test.reshape(X_test.shape[0], -1)
#     else:
#         segments, labels = segmentation_prediction(X, y, window, 0.2)
#         X_train, X_test, y_train, y_test = split_data_balanced(segments, labels, test_size=test_size//100)
#         # Reshape the feature matrices for SVM
#         X_train = X_train.reshape(X_train.shape[0], -1)
#         X_test = X_test.reshape(X_test.shape[0], -1)
#     return X_test, y_test
# def prepare_data_LSTM(df, Features, target, p,window_size,test_size):
#     save_path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\normalization_params_LSTM.json'
#     with open(save_path, 'r') as f:
#         normalization_params = json.load(f)
#     loaded_min = np.array(normalization_params['min'])
#     loaded_max = np.array(normalization_params['max'])
#     X = df[Features].values
#     y = df[target].values
#     X = (X - loaded_min) / (loaded_max - loaded_min)
#     if p == 'Detection':
#         segments, labels = segmantation(X, y, window_length=60)
#     else:
#         segments, labels = segmentation_prediction(X, y, 60, 0.2)
#     print('seg', segments.shape)
#     X_test, y_test = np.array(segments), np.array(labels).reshape(- 1, 1)
#     print(f'x.shape {X_test.shape}, \n ytest.shape: {y_test.shape}')
#     return X_test, y_test
    

# def load_models_prediction(df, featurs, target, option, detect_predict,window):

#     if option == 'Support Vector Machine (SVM) ':
#         if detect_predict == 'Detection':
#             path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\svm_detection_kernel_rbf_c_10_gamma_1__precision_0.993_recall_0.912_fscore_0.951_window_60.pkl'
#         else:
#             path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\svm_prediction_kernel_rbf_c_10_gamma_1__precision_1.0_recall_0.235_fscore_0.381_window_60.pkl'
#         with open(path, "rb") as f:
#             model = pickle.load(f)
#             # Make a prediction.
#         X, y = prepare_data_svm_random_forest(df, featurs, target,detect_predict,window)
#         prediction = model.predict(X)

#         # Print the prediction.
#         return y,prediction
#     if option == 'LSTM Long-Short-Term-Memory (LSTM)':
#         if detect_predict == 'Detection':
#             path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\model_LSTM_detection_epoch300_batch_16,precision_0.993_recall_0.84_fscore_0.91_window_60.h5'
#         else:
#             path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\LSTM_prediction_epoch300_batch_64,precision_0.0_recall_0.0_fscore_0.0_window_60.h5'
#         model = tf.keras.models.load_model(path)
#         X, y = prepare_data_LSTM(df, featurs, target,detect_predict)
#         prediction = model.predict(X)
#         y_pred = (prediction >= 0.5).astype(int)
#         print('qfgqsgsrgsgsdg',y.shape)
#         print('sdfgsdfbsdfbsdfbsdfbsdfbsdfbsdfb',y_pred.shape)
#         return  y, y_pred
        
        
#     if option == 'Random Forest':
#         if detect_predict == 'Detection':
#             path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\model_Random_forest_detection_est_200_depth_10_split_2_leaf_1_precision_1.0_recall_0.966_fscore_0.982_window=60.pkl'
#         else:
#             path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\model_Rndom_forest_prediction_est_100_depth_10_split_2_leaf_1_precision_1.0_recall_0.412_fscore_0.583_window60.pkl'
#         with open(path, "rb") as f:
#             model = pickle.load(f)
#             # Make a prediction.
#         X, y = prepare_data_svm_random_forest(df, featurs, target,detect_predict,window)
#         print(X.shape)
#         prediction = model.predict(X)
#         return y,prediction
        