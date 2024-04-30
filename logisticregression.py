import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pandas as pd


one_hot_encode_feat = []
otherfeat = []

df = pd.read_csv("C:/Users/thoma/OneDrive/Desktop/CS365 Data/dataset_picked_champions_players_statistics.csv")

df = df.iloc[:,1:]

for i in range (df.shape[1]):
    columnName = df.columns[i]
    if columnName != 'result':
        if (i-1)%4 == 0:
            one_hot_encode_feat.append(columnName)
        else:
            otherfeat.append(columnName)
        
XALL = one_hot_encode_feat + otherfeat


train, test = train_test_split(df, test_size=.2, train_size=.8, random_state=None, shuffle=True, stratify=None)
    
X = train.loc[:, XALL]
y = train.result


pipe = make_pipeline(
   ColumnTransformer(
       transformers=[
        ("one-hot-encode", OneHotEncoder(handle_unknown='ignore'), one_hot_encode_feat),
       ],
       
        remainder="passthrough",
   ),


   LogisticRegression(solver = 'newton-cholesky', penalty = 'l2'),
    

)

# Fit the pipeline to your data
pipe.fit(X, y)


X_new = test.loc[:, XALL]
new_pred_class = pipe.predict(X_new)


print('Training accuracy: ', accuracy_score(train.result, pipe.predict(X)))

print('Test accuracy: ', accuracy_score(test.result, new_pred_class))

#features importance:

logistic_regression_model = pipe.named_steps['logisticregression']

coefficients = logistic_regression_model.coef_[0]

# If you have a simple feature setup without transformations that obscure feature names:
features = XALL  # This should be expanded to include all features after encoding

# Print the feature importance
importance = zip(features, coefficients)
sorted_importance = sorted(importance, key=lambda x: x[1], reverse=True)
for feature, coef in sorted_importance:
    print(f"Feature: {feature}, Coefficient: {coef}")

