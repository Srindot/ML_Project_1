from sklearn.impute import SimpleImputer
import pandas as pd

def handle_missing_values(dataframe, strategy="median"):
    imputer = SimpleImputer(strategy=strategy)
    housing_num = dataframe.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    return housing_tr


