import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.dataset = None
        self.sensor = None
        self.target = None

    def load(self,file,isTest=False):
        if not isTest:
            print("loading")
            self.dataset = pd.read_csv(file)
            print("loaded")
            self.target = self.dataset[['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']]
            self.sensor = self.dataset[list(self.dataset.columns[1:-3])]
            self.sensor.columns = ['deg_C', 'rel_h', 'abs_h','s_1', 's_2', 's_3', 's_4', 's_5']
        else:
            print("loading test files")
            self.dataset = pd.read_csv(file)
            print("Test files loaded")
            self.sensor = self.dataset[list(self.dataset.columns[1:])]
            self.sensor.columns = ['deg_C', 'rel_h', 'abs_h', 's_1', 's_2', 's_3', 's_4', 's_5']

    def process(self):
        col = ['s_1', 's_2', 's_3', 's_4', 's_5']
        df = self.sensor[col].values/1000
        self.sensor.update(pd.DataFrame(df))


    def split(self):
        if type(self.target) == None:
            print("This is test data")
            return
        X_train, X_test, y_train, y_test = \
            train_test_split(self.sensor, self.target, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

