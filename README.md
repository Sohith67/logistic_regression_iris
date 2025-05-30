Logistic Regression Binary Classifier – Iris Dataset 🌸
📌 Objective
This project demonstrates binary classification using Logistic Regression on a subset of the Iris dataset. The goal is to:

Preprocess data

Train a logistic regression model

Evaluate its performance using standard classification metrics

Visualize the sigmoid function and ROC curve

📊 Dataset
Source: Scikit-learn Iris dataset
We use only two classes (Setosa and Versicolor) from the original dataset to create a binary classification problem.

🧰 Tools & Libraries
Python
Scikit-learn
Pandas
NumPy
Matplotlib

🧪 Steps Performed
Data Loading
Loaded the Iris dataset and filtered it to keep only two classes for binary classification.

Train/Test Split
Split the dataset using train_test_split (70% training, 30% testing).

Feature Standardization
Standardized the features using StandardScaler.

Model Training
Trained a LogisticRegression model on the training data.

Evaluation Metrics

Confusion Matrix

Precision

Recall

ROC-AUC Score

ROC Curve Plot

Sigmoid Function Visualization
Plotted the sigmoid curve to demonstrate how logistic regression maps outputs between 0 and 1.

📈 Results
The model successfully classified the two Iris species.

ROC-AUC score and precision/recall metrics provided insight into the model's performance.

Sigmoid function and ROC curve were visualized for better conceptual understanding.

