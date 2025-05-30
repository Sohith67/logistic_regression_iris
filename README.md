Binary Classification on Iris Dataset Using Logistic Regression
This project applies Logistic Regression to the classic Iris dataset, converting the multi-class classification problem into a binary classification problem (Iris-setosa vs Non-Setosa). The model is built using scikit-learn and includes data preprocessing, feature selection, regularization, stratified cross-validation, and learning curve visualization.


📊 Dataset
Source: Iris Dataset - UCI ML Repository

Original Target Classes: Iris-setosa, Iris-versicolor, Iris-virginica

Modified Target:

1 → Iris-setosa

0 → Iris-versicolor or Iris-virginica

🛠️ Tools & Libraries
Python

pandas

numpy

matplotlib

seaborn

scikit-learn

🧪 Workflow
Data Loading & Cleaning

Load CSV

Drop ID column

Binary target conversion

Train-Test Split

Stratified 80/20 split to maintain class balance

Feature Selection

SelectKBest using ANOVA F-test (top 2 features)

Preprocessing

StandardScaler for normalization

Model Training

LogisticRegression with:

C=0.001 for stronger regularization

class_weight='balanced'

Evaluation

Confusion matrix

Classification report

Stratified 5-fold cross-validation

Learning curve plotting

📈 Output Example
Classification Report:

![image](https://github.com/user-attachments/assets/80b187a8-5406-4665-a577-ff71e6a9a0db)

The learning curve shows how training and validation accuracy change with training size — useful to check overfitting/underfitting.

▶️ How to Run
Clone the repository:


git clone https://github.com/yourusername/iris-logistic-regression.git
cd iris-logistic-regression
Install dependencies:


Run the script:

python iris_logistic_regression.py

