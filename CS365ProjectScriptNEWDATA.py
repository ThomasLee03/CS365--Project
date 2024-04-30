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
       #    ("one-hot-encode", OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_encode_feat),
        ],
       
        remainder="passthrough",
   ),
 #  PCA(n_components=201),
  #  CategoricalNB(),
  #sklearn.ensemble.RandomForestClassifier(n_estimators = 100, criterion = "log_loss")
  #naive_bayes.MultinomialNB()
  #naive_bayes.CategoricalNB()
   LogisticRegression(solver = 'newton-cholesky', penalty = 'l2'),
   #tree.DecisionTreeClassifier(),
    

)

#feature_importances_ 
# Fit the pipeline to your data
pipe.fit(X, y)

# Access the LogisticRegression model from the pipeline
#logistic_regression_model = pipe.named_steps['logisticregression']

# Access the coefficients
#coefficients = logistic_regression_model.coef_

#print("Coefficients:", coefficients)



X_new = test.loc[:, XALL]
new_pred_class = pipe.predict(X_new)


print('Training accuracy: ', accuracy_score(train.result, pipe.predict(X)))

print('Test accuracy: ', accuracy_score(test.result, new_pred_class))

print(confusion_matrix(test.result, new_pred_class))
logistic_regression_model = pipe.named_steps['logisticregression']
# Getting the coefficients
coefficients = logistic_regression_model.coef_[0]

# If you have a simple feature setup without transformations that obscure feature names:
features = XALL  # This should be expanded to include all features after encoding

# Print the feature importance
importance = zip(features, coefficients)
sorted_importance = sorted(importance, key=lambda x: x[1], reverse=True)
for feature, coef in sorted_importance:
    print(f"Feature: {feature}, Coefficient: {coef}")


#OHE + REG

#Training accuracy:  0.8569542253521126
#Test accuracy:  0.846830985915493

#Characters picked 

#Training accuracy:  0.772887323943662
#Test accuracy:  0.6373239436619719

#Without OHE

#Training accuracy:  0.8257042253521126
#Test accuracy:  0.801056338028169

#without regularization:
    
#Training accuracy:  0.8538732394366197
#Test accuracy:  0.8221830985915493

#LBFGS failed to converge 

#NewtonCholesky with l2 

#Training accuracy:  0.9375
#Test accuracy:  0.8838028169014085

#categorical NB

#Training accuracy:  0.9036091549295775
#Test accuracy:  0.8626760563380281

#random forest:
    
#Training accuracy:  1.0
#Test accuracy:  0.8767605633802817    


#decision tree:
    
#Training accuracy:  1.0
#Test accuracy:  0.7887323943661971

#pca components: around 201 (arbitarily picked from 1 - 500 with increments of 50)

