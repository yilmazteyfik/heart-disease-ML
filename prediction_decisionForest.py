import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset (replace 'file_path' with your file's path)
file_path = "heart.csv"
heart_data = pd.read_csv(file_path)

# Project Objective: To build a machine learning model for diagnosing heart disease.
print("Project Objective: Building a machine learning model for heart disease diagnosis.")

# Display the first few rows of the dataset
print(heart_data.head())

# Check for missing values
print(heart_data.isnull().sum())

# Calculate correlation matrix
correlation_matrix = heart_data.corr()

# Find highly correlated features
highly_correlated = (correlation_matrix.abs() > 0.5) & (correlation_matrix.abs() < 1.0)

# Extract the correlated features
correlated_features = set()
for i in range(len(highly_correlated.columns)):
    for j in range(i):
        if highly_correlated.iloc[i, j]:
            colname = highly_correlated.columns[i]
            correlated_features.add(colname)

print("Highly correlated features:")
print(correlated_features)

# Separating features and target variable
X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handling missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Initializing and training the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_imputed, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test_imputed)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy}")
