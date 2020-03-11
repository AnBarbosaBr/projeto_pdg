import pandas as pd 
import sklearn.model_selection

SEED = 42
TRAIN_SIZE = 0.7

data = pd.read_csv("../data/car_insurance_claim.csv")

train_data, test_data = sklearn.model_selection.train_test_split(data, train_size = TRAIN_SIZE, random_state = SEED)

train_data.to_csv("../data/car_insurance_clain_train.csv", index = False)
test_data.to_csv("../data/car_insurance_clain_test.csv", index = False)


    