import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

print(housedata.columns)
print(housedata.describe())

X = housedata.drop('unitprice', axis=1)
y = housedata['unitprice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=666)

RF = RandomForestRegressor(n_estimators=100)
RF.fit(X_train, y_train)

print(RF)
pred_train = RF.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
mre_train = mean_absolute_error(y_train, pred_train) / y_train.mean()
print(1 - mse_train / y_train.var())
print(mse_train)
print(mre_train)

pred_test = RF.predict(X_test)
mse_test = mean_squared_error(y_test, pred_test)
mre_test = mean_absolute_error(y_test, pred_test) / y_test.mean()
print(1 - mse_test / y_test.var())
print(mse_test)
print(mre_test)

print(pd.concat([y_train, pd.Series(pred_train), abs(pred_train - y_train)], axis=1).head(10))
print(pd.concat([y_test, pd.Series(pred_test), abs(pred_test - y_test)], axis=1).head(10))

print(RF.estimators_[0].mse())
RF.estimators_[0].plot()

MRE_train = []
MRE_test = []
for t in range(2, 21):
    RF1 = RandomForestRegressor(n_estimators=100)
    RF1.fit(X_train, y_train)
    pred_train1 = RF1.predict(X_train)
    pred_test1 = RF1.predict(X_test)
    mse_train1 = mean_squared_error(y_train, pred_train1)
    mse_test1 = mean_squared_error(y_test, pred_test1)
    mre_train1 = mean_absolute_error(y_train, pred_train1) / y_train.mean()
    mre_test1 = mean_absolute_error(y_test, pred_test1) / y_test.mean()
    MRE_train.append(mre_train1)
    MRE_test.append(mre_test1)

print(MRE_train)
print(MRE_test)