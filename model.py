import pandas as pd
import numpy as np
from dataHandler import DataLoader
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor

def linear(X,X_,y,y_):
    clf = Ridge(normalize=True).fit(X, y)
    pred = np.array(clf.predict(X_))
    pred = np.around(pred, 1).tolist()
    print(pred[:20])
    print(np.around(y_.values[:20], 1).tolist())
    print(clf.score(X_, y_))

    reg = LinearRegression(normalize=True).fit(X,y)
    lpred = np.array(reg.predict(X_))
    lpred = np.around(lpred, 1).tolist()
    print(lpred[:20])
    print(np.around(y_.values[:20], 1).tolist())
    print(reg.score(X_, y_))


def randomForest(X,y):
    max_depth = 25
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=45))
    clf = regr_multirf.fit(X,y)
    return clf


if __name__ == '__main__':
    dl = DataLoader()
    dl.load("datasets/train.csv")
    X, X_, y, y_ = dl.split()
    dl.process()
    reg = randomForest(X,y)
    tdl = DataLoader()
    tdl.load("datasets/test.csv",isTest=True)
    tdl.process()
    pred = np.array( reg.predict(tdl.sensor))
    date = tdl.dataset.date_time.to_numpy().reshape(-1,1)
    data = np.concatenate((date,pred),axis=1)
    result = pd.DataFrame(data,columns=["date_time","target_carbon_monoxide","target_benzene","target_nitrogen_oxides"])
    result.to_csv("datasets/result.csv",index=False)

