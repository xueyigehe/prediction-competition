#--**coding:utf-8**--
import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import Lasso,LinearRegression,LassoCV
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,scale,MaxAbsScaler

data = pd.read_csv('tap_fun_train.csv',parse_dates = True)
data_train = copy.copy(data[data['avg_online_minutes'] >= 12]) #分析了训练数据情况，处理平均在线时长小于5min的，可预测45天的就为7天内的付费值
print data_train.shape

data_train = data_train.drop(['user_id','register_time'],axis=1)
print data_train.shape

x_train = data_train.loc[:, data_train.columns != 'prediction_pay_price']
y_train = data_train.loc[:, data_train.columns == 'prediction_pay_price']

model = Lasso() #分析了训练数据，存在大量共线，可使用L1正则化消除共线
model.fit(x_train, y_train)
print model.coef_
print len(model.coef_)

none_mean = []
for i in range(len(model.coef_)):
    if abs(model.coef_[i]) < 1e-06:
        none_mean.append(x_train.columns[i])
        
print none_mean
print len(none_mean)

x_train_final = x_train.drop(none_mean, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x_train_final,y_train,test_size = 0.2, random_state = 0)
X_test_7pays = X_test['pay_price'].tolist()

scaler = StandardScaler(with_mean=False).fit(X_train.values)
#scaler = MaxAbsScaler().fit(X_train.values)
X_train = scaler.transform(X_train.values)
X_test = scaler.transform(X_test.values)

SGDModel = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01)#RandomForestRegressor(n_estimators=500)#LinearRegression()##SGDRegressor()
#可使用网格法选参数，然而分出的验证集并不好，可使用k折交叉验证试试
SGDModel.fit(X_train, Y_train.values.ravel())
joblib.dump(SGDModel,'train_model.m')

y_pred = SGDModel.predict(X_test)
print Y_train.values
for i in range(len(y_pred)):
    if y_pred[i] < X_test_7pays[i]: #判断45天的预测值若比前7天的还小，则使用前7天的付费金额作为预测值
        y_pred[i] = X_test_7pays[i]

for i in range(len(y_pred)):
    if y_pred[i] < 0:
        y_pred[i] = 0
#     elif y_pred[i] < 1.4:
#         y_pred[i] = 0.99
print("Root Mean squared error: %.2f"
      % mean_squared_error(Y_test, y_pred) ** 0.5)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, y_pred))

data1 = pd.read_csv('tap_fun_test.csv',parse_dates=True)
print data1.shape
data_test1 = copy.copy(data1[data1['avg_online_minutes'] < 12])
data_test2 = copy.copy(data1[data1['avg_online_minutes'] >= 12])
data_test2_7pays = data_test2['pay_price'].tolist()

#看了训练数据情况，处理平均在线时长小于5min的，可预测45天的就为7天内的付费值
data_test_part1 = data_test1[['user_id','pay_price']]
data_test_part1.rename(columns={'pay_price':'prediction_pay_price'},inplace = True)
data_test_part1.to_csv('tap_fun_result_part1.csv')

data_test2_id = data_test2['user_id'].values
data_test2_id_final = pd.DataFrame(data_test2_id,columns={'user_id'})
data_test2 = data_test2.drop(['user_id','register_time'],axis=1)

data_test2_final = data_test2.drop(none_mean, axis=1)
data_test2_final = scaler.transform(data_test2_final.values)
data_test2_pred = SGDModel.predict(data_test2_final)

for i in range(len(data_test2_pred)):
    if data_test2_pred[i] < data_test2_7pays[i]:
        data_test2_pred[i] = data_test2_7pays[i]

for i in range(len(data_test2_pred)):
    if data_test2_pred[i] < 0:
        data_test2_pred[i] = 0
#     elif data_test2_pred[i] < 1.4:
#         data_test2_pred[i] = 0.99

data_test2_pred_howmuch = pd.DataFrame(data_test2_pred,columns={'prediction_pay_price'})

data_test2_result = pd.concat([data_test2_id_final,data_test2_pred_howmuch],axis=1)
data_test2_result.to_csv('tap_fun_result_part2.csv')

pred_part1 = pd.read_csv('tap_fun_result_part1.csv',index_col=0,parse_dates=True)
pred_part2 = pd.read_csv('tap_fun_result_part2.csv',index_col=0,parse_dates=True)

pred = pred_part1.append(pred_part2)
print pred.shape
pred.to_csv('pred_result.csv',index=False)


