import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB

# Load dataset
df = pd.read_csv("C:/Users/thoma/OneDrive/Desktop/CS365 Data/dataset_picked_champions_players_statistics.csv")
df = df.iloc[:, 1:]  # assuming the first column is not needed

# Identify categorical features (excluding the target variable 'result')
categorical_features = [col for col in df.columns if col != 'result']

# Splitting the dataset
train, test = train_test_split(df, test_size=0.2, random_state=42)  # Better to set a random_state for reproducibility
X_train = train[categorical_features]
y_train = train['result']
X_test = test[categorical_features]
y_test = test['result']


clf = CategoricalNB()
clf.fit(X_train, y_train)


# 4. Calculate training and testing accuracies
train_accuracy = accuracy_score(y_train, clf.predict(X_train))
test_accuracy = accuracy_score(y_test, clf.predict(X_test))

# 5. Print out the training and testing accuracies
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


