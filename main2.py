from fetch_data import fetch_housing_data
from load_data import load_housing_data
from Categorize_Income import CategorizeIncome
from StratificationData import StratifiedSampling
from RemoveIncomeCat import RemoveIcomeCat
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error

import numpy as np

# Fetching the data from the net
# fetch_housing_data()

# Fetch the data
housing_data = load_housing_data()      

# Creating an Income Category 
strat_housing_data = CategorizeIncome(housing_data)

# Splitting the stratified data into training and testing
strat_trainSet, strat_testSet = StratifiedSampling(strat_housing_data, "IncomeCategory")

# Removing the income category after usage
strat_train, strat_test = RemoveIcomeCat(strat_trainSet, strat_testSet)

# Creating the training set and the labels 
housing = strat_train.drop("median_house_value", axis=1)
housing_labels = strat_train["median_house_value"].copy()

# Data cleaning 
# Removing the text attribute
housing_num = housing.drop("ocean_proximity", axis=1)

# Numerical and categorical attributes
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# Define numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Define full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# Prepare the housing data
housing_prepared = full_pipeline.fit_transform(housing)

# Selecting a model and training
reg = DecisionTreeRegressor()  # Changed to DecisionTreeRegressor
reg.fit(housing_prepared, housing_labels)

# Make predictions on the training data
housing_predictions = reg.predict(housing_prepared)

# Calculate RMSE on training data
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

print("RMSE = ", rmse)
