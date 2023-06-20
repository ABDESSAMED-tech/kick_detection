# The Application of Machine Learning in Kick Detection and Prediction During Drilling Operations

This Streamlit application provides a comprehensive solution for kick detection and prediction during drilling operations using machine learning algorithms. The application consists of six pages, each serving a specific purpose.

## 1. Load the Dataset
In this page, you can upload your drilling operations dataset. Ensure that the dataset is in a compatible format for processing.

## 2. Get Summary of Dataset
This page allows you to view a summary of the loaded dataset, including statistical measures, data distribution, and other relevant information.

## 3. Application of Algorithm Approach: Variate Thresholds
Explore the impact of different thresholds on kick detection and prediction. Adjust the thresholds to observe how they affect the algorithm's performance.

## 4. The Application of LSTM, SVM, and Random Forest: Variate Hyperparameters and Save the Models
This page focuses on training and evaluating machine learning models, specifically Long Short-Term Memory (LSTM), Support Vector Machines (SVM), and Random Forest. Experiment with different hyperparameters to improve model performance and save the trained models for future use.

## 5. Visualize the Dataset: Scatter, Subplot, Correlation Matrix
Visualize the dataset through scatter plots, subplots, and correlation matrices. Gain insights into the relationships between variables and identify patterns within the data.

## 6. Use Saved Models for Prediction or Detection
Utilize the previously saved machine learning models to make predictions or detect kicks in real-time data. Provide new data inputs and observe the model's output for detection or prediction purposes.

To run this Streamlit application locally, follow these steps:
1. Install the necessary dependencies by running `pip install -r requirements.txt`.
2. the data set  in DATASET folder.
3. Execute the application by running `streamlit run Load data.py`.
4. Access the application by opening the provided local URL in your web browser.

