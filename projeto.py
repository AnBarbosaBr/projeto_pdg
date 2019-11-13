
import pandas as pd;
import numpy as np; 
from sklearn.tree import DecisionTreeClassifier;



 

# OBS: Scikit-Learn
# scikit-learn uses an optimised version of the CART algorithm; however, scikit-learn implementation does not support categorical variables for now.
# https://scikit-learn.org/stable/modules/tree.html


data = pd.read_csv("data/car_insurance_claim.csv")

# Linhas / Colunas
data.shape 

pd.DataFrame(data.columns)



data.describe()
data.describe().shape[1] # Colunas Númericas


# Separate Data in Training and Test
independent_variables = list(range(1,25))
dependent_variables = 25

training_data = data[0:900]
test_data = data[901:]


x_training = training_data.iloc[ : , independent_variables]
y_training = training_data.iloc[ : , dependent_variables]
x_test = test_data.iloc[ : , independent_variables]
y_test = test_data.iloc[ : , dependent_variables]

# GLM
## Setup

## Train
## Test

# Decision Tree - CART
## Setup
cartTree = DecisionTreeClassifier()


## Train
cartTree.fit(x_training, y_training)
train_tree_predictions = cartTree.predict(x_training)

## Test
train_tree_predictions = cartTree.predict(x_test)

# RBFN
## Setup

## Train
## Test


# Compare results



