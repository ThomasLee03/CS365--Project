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

import pandas as pd

df = pd.read_csv("C:/Users/thoma/OneDrive/Desktop/CS365 Data/Hello.csv")

train, test = train_test_split(df, test_size=.2, train_size=.8, random_state=None, shuffle=True, stratify=None)


feature_cols = ['Champion1', 'Champion2','Champion3','Champion4','Champion5','Champion6','Champion7','Champion8','Champion9','Champion10']
X = train.loc[:, feature_cols]
y = train.WinFirstTeam


#model = LogisticRegression(solver = 'newton-cholesky', penalty = 'l2').fit(X, y)

pipe = make_pipeline(
    ColumnTransformer(
       transformers=[
           ("one-hot-encode", OneHotEncoder(), feature_cols),
        ],
        remainder="passthrough",
   ),
    LogisticRegression(penalty = 'l2'),
  #MultinomNB()
  #LogisticRegresialsion(penalty = 'l2')
  #  naive_bayes.CategoricalNB()
 #   tree.DecisionTreeClassifier(max_depth = 6)
)


#pipe.fit(X, y)

# Fit the pipeline to your data
pipe.fit(X, y)

# Access the LogisticRegression model from the pipeline
logistic_regression_model = pipe.named_steps['logisticregression']

# Access the coefficients
coefficients = logistic_regression_model.coef_

print("Coefficients:", coefficients)



#these are the weights
#print(pipe.coef_)


#import seaborn as sns
#sns.regplot(x='target', y='variable', data=data, logistic=True)






X_new = test.loc[:, feature_cols]
new_pred_class = pipe.predict(X_new)


print('Training accuracy: ', accuracy_score(train.WinFirstTeam, pipe.predict(X)))

print('Test accuracy: ', accuracy_score(test.WinFirstTeam, new_pred_class))

print(confusion_matrix(test.WinFirstTeam, new_pred_class))

