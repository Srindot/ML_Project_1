import pandas as pd

def RemoveIcomeCat(strat_train, strat_test):
    for set in (strat_train, strat_test):
        set.drop("IncomeCategory", axis=1, inplace=True)
    return strat_train, strat_test