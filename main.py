from fetch_data import fetch_housing_data
from load_data import load_housing_data
from create_testcase import spllit_train_test
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pandas.plotting import scatter_matrix
from cleaningdata import handle_missing_values
from Categorize_Income import CategorizeIncome
from StratificationData import StratifiedSampling
from transformers import DataFrameSelector
from transformers import CombinedAttributesAdder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

ordinal_encoder = OrdinalEncoder()
cat_encoder = OneHotEncoder()
lin_reg = LinearRegression()

#fetch_housing_data() #Run to Fetch the data
housing_data = load_housing_data()
print("Data = \n")
print(housing_data.head())
print("\n******************************************************\n")
print(housing_data.info())
print("\n******************************************************\n")
print(housing_data["ocean_proximity"].value_counts())
print("\n******************************************************\n")
print(housing_data.describe())

#matplot histograph representation
housing_data.hist(bins = 50)
plt.show()

print("\n******************************************************\n")
print("Splitting the data for training and testing purposes")
"""train_set, test_set = spllit_train_test(housing_data, 0.2)
print("size of training set = ", len(train_set))
print("size of testing set = ", len(test_set))"""
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
print("size of training set = ", len(train_set))
print("size of testing set = ", len(test_set))
print("\n******************************************************\n")
housing_data_cat = CategorizeIncome(housing_data)
housing_data_cat["IncomeCategory"].hist()
plt.show()

print("\n******************************************************\n")
strat_train_set, strat_test_set = StratifiedSampling(housing_data_cat, "IncomeCategory")
print("Printing the stratified data ")
print("strat test data = ",strat_test_set["IncomeCategory"].value_counts() / len(strat_test_set))


print("\n******************************************************\n")
print("Removing the income category attribute from the stratified training set")
for set_ in (strat_train_set, strat_test_set):
    set_.drop("IncomeCategory", axis=1, inplace=True)
print("Removing done")

print("\n******************************************************\n")
#visualize the data
#copying the data 
housing = strat_train_set.copy()
housing.plot(kind = "scatter", x = "longitude", y = "latitude")
plt.show()

print("\n******************************************************\n")

print("Decreasing the alpha value")
housing.plot(kind = "scatter", x = "longitude", y = "latitude",alpha = 0.1)
plt.show()

print("\n******************************************************\n")
print("using color map")
housing.plot(kind = "scatter", x = "longitude", y = "latitude",alpha = 0.1, s = housing["population"]/200, label = "population",c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
plt.legend()
plt.show()

print("\n******************************************************\n")
print("Looking for Correlation")
"""corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)"""

numeric_columns = housing.select_dtypes(include=[np.number]).columns
corr_matrix = housing[numeric_columns].corr()
print("Correlation with median_house_value:")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

print("\n******************************************************\n")
print("Printing the correlation grpahs")
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize = (12, 8))
plt.show()
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)
plt.show()

# Feature adding attributes
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]


# processing the data 

# Handle missing values
housing_tr = handle_missing_values(housing)

# Handle categorical attributes
housing_cat = housing[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

# Get feature names for categorical data
cat_feature_names = list(cat_encoder.get_feature_names_out(['ocean_proximity']))
housing_cat_df = pd.DataFrame(housing_cat_1hot.toarray(), columns=cat_feature_names)

# Concatenate numerical and categorical data
housing_prepared = pd.concat([pd.DataFrame(housing_tr, columns=housing_tr.columns), housing_cat_df], axis=1)

print("Prepared Housing Data:\n", housing_prepared.head())


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Training a model")

housing_labels = strat_train_set["median_house_value"].copy()
lin_reg.fit(housing_prepared, housing_labels)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#taking some data and testing it 
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = pd.concat([pd.DataFrame(handle_missing_values(some_data), columns=some_data.columns), pd.DataFrame(cat_encoder.transform(ordinal_encoder.transform(some_data[["ocean_proximity"]]).reshape(-1, 1)).toarray(), columns=cat_feature_names)], axis=1)

# Predict 
housing_predictions = lin_reg.predict(housing_prepared)

# Calculate Mean Squared Error (MSE) or Root Mean Squared Error (RMSE)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE) on training set: {rmse}")
