import pandas as pd 
import numpy as np
def CategorizeIncome(housing_data):
    housing_data["IncomeCategory"] = pd.cut(housing_data["median_income"],bins = [0., 1.5, 3.0, 4.5, 6., np.inf],labels = [1, 2, 3, 4, 5])
    return housing_data