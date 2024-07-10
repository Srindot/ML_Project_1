from sklearn.model_selection import StratifiedShuffleSplit
def trainTestSplit(housingdata, ratio):
    trainSet, testSet = train_test_split(housingdata, test_size = ratio, random_state = 42)
    return trainSet, testSet
