# Heart Disease Prediction using Machine Learning and Deep Learning

## About
This project focuses on predicting heart disease using both traditional machine learning techniques and deep learning models. It also includes converting the trained model to TensorFlow Lite format for deployment on mobile devices, such as Android applications. The project leverages a publicly available dataset from Kaggle to train models that can identify the presence of heart disease in patients based on various health metrics.

## Dataset
The dataset used in this project is obtained from Kaggle:
[Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

- **Columns:** 
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
  - `target` (0: No heart disease, 1: Heart disease)

## Project Overview
### Data Preprocessing
- **Feature Scaling:** Standardized the features using `StandardScaler`.
- **Train-Test Split:** Split the data into 80% training and 20% testing sets.

### Models Implemented
1. **Random Forest Classifier:**
   - Used to classify the presence of heart disease.
   - Achieved high accuracy and balanced precision and recall.

2. **Deep Neural Network:**
   - Implemented using TensorFlow to predict heart disease.
   - The model consists of two hidden layers with ReLU activation and a sigmoid output layer.

### Model Evaluation
- **Random Forest Classifier:**
  - Accuracy: 83.61%
  - Precision: 85.11%
  - Recall: 84.57%
  - F1-score: 84.83%
  
- **Neural Network Model:**
  - Trained for 50 epochs with a learning rate of 0.001.
  - Final accuracy is evaluated on the test set.

### TensorFlow Lite Conversion
- Converted the trained deep learning model to TensorFlow Lite format, making it suitable for deployment on edge devices, such as Android applications.

### Libraries Used
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computing.
- **Scikit-learn:** For implementing the Random Forest Classifier and performing data preprocessing tasks like feature scaling and train-test splitting.
- **TensorFlow:** For building and training the deep neural network model.
- **Matplotlib and Seaborn:** For data visualization, including plotting confusion matrices and comparing model performance.
- **Joblib:** For model serialization and saving.
- **TensorFlow Lite:** For converting the trained TensorFlow model into a format suitable for deployment on mobile devices.

### Deployment on Android
- The TensorFlow Lite model (`heart_disease_model.tflite`) can be integrated into an Android application to predict heart disease in real-time. This integration allows users to input health metrics directly through the app and receive immediate feedback on their heart health status.
