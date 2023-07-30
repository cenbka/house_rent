#Загружаем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
from sklearn.ensemble import RandomForestRegressor
color = sns.color_palette()
#%matplotlib inline
training_csv = pd.read_csv(r'C:\Users\Арсений\PycharmProjects\diplom\train.csv')

#Нормализация
for t in training_csv.columns:
    if training_csv[t].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(training_csv[t].values))
        training_csv[t] = lbl.transform(list(training_csv[t].values))
trainX = training_csv.drop(["id", "timestamp", "price_doc"], axis=1)
trainY = training_csv["price_doc"]
#Загружаем метод разделения выборки
from sklearn.model_selection import train_test_split

#Готовим подвыборки
xTrain, xTest, yTrain, yTest = train_test_split(trainX, trainY, test_size=0.4)

#Подгружаем RMSLE
from sklearn.metrics import mean_squared_log_error

#Замена Nan на среднее
#Xtrain = Xtrain.fillna(0)
#Ytrain = Ytrain.fillna(0)
#Xtest = Xtest.fillna(0)
#Ytest = Ytest.fillna(0)
xTrain = xTrain.fillna(xTrain.mean())
yTrain = yTrain.fillna(yTrain.mean())
xTest = xTest.fillna(xTest.mean())
yTest = yTest.fillna(yTest.mean())


#Организуем обучение в цикле для Random Forest
gr = np.arange(1, 15, 1)
facc = []
acc = 0
for i in gr:
    scc = 0
    model = RandomForestRegressor(n_estimators=i, max_depth = 8)
    model.fit(xTrain, yTrain)
    y_predicted = model.predict(xTest)
    scc = mean_squared_log_error(yTest, y_predicted)
    facc.append(scc)
    if scc < acc:
        acc = scc
        mf = i
    elif i < 2:
        acc = scc
    print("Random Forest: , n_estimators", i, " Функция потерь RMSLE", scc)
plt.plot(gr, facc)
plt.title("Точность модели в зависимости от числа деревьев")
plt.xlabel("Число деревьев")
plt.ylabel("Функция потерь RMSLE")
plt.show()
print("best n_estimators", mf, "Наименьшая функция потерь RMSLE", acc)
#scc=mean_squared_log_error(Ytest, y_predicted)
#print("Error RMSLE", scc)
gmd=np.arange(1,16,1)
facc_md=[]
acc=0
for i in gmd:
    scc = 0
    model = RandomForestRegressor(n_estimators=mf, max_depth=i)
    model.fit(xTrain, yTrain)
    y_predicted = model.predict(xTest)
    scc = mean_squared_log_error(yTest, y_predicted)
    facc_md.append(scc)
    if scc < acc:
        acc = scc
        mf = i
    elif i < 2:
        acc = scc
    print("Random Forest: , Max_depth", i, " Функция потерь RMSLE", scc)
plt.plot(gmd, facc_md)
plt.title("Точность модели в зависимости от глубины деревьев")
plt.xlabel("Глубина деревьев")
plt.ylabel("Функция потерь RMSLE")
plt.show()
print("best Max_depth", mf, "Наименьшая функция потерь RMSLE", acc)
#Определение обучающей матрицы
dtrain = xgb.DMatrix(xTrain, yTrain)
#Определение тестовой матрицы
dtest = xgb.DMatrix(xTest)
#Цикл для XGB eta
i = 0.01
acc_xgb = 0
gmd = np.arange(1, 16, 1)
gr = np.arange(0.01, 0.1, 0.01)
err = []
for et in gr:
    xgb_params = {
     'eta': et,
     'max_depth': 8,
     'subsample': 0.7,
     'colsample_bytree': 0.7,
     'objective': 'reg:linear',
     'eval_metric': 'rmse',
     'verbosity': 1
    }
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=100)
    tt = model.predict(dtest)
    scc_xgb = mean_squared_log_error(yTest, tt)
    print("eta", et, "i", i, "scc_xgb", scc_xgb)
    if scc_xgb > acc_xgb:
        acc_xgb = scc_xgb
        best_i = i
    err.append(scc_xgb)
plt.plot(gr, err)
plt.title("Зависимость значения погрешности модели от скорости обучения")
plt.xlabel("eta, скорость обучения")
plt.ylabel("Функция потерь RMSLE")
plt.show()
acc_xgb = 0
gr=np.arange(0.01,0.1,0.01)
#gmd= np.arange(1,16,1)
err = []
err_x = []
for imd in gmd:
    print("imd", imd, "i", i, "scc_xgb", scc_xgb)
    xgb_params = {
      'eta': 0.03,
      'max_depth': imd,
      'subsample': 0.7,
      'colsample_bytree': 0.7,
       'objective': 'reg:linear',
       'eval_metric': 'rmse',
       'verbosity': 1
    }
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=100)
    tt = model.predict(dtest)
    scc_xgb = mean_squared_log_error(yTest, tt)
    if scc_xgb > acc_xgb:
        acc_xgb = scc_xgb
        best_i = i
    err_x.append(scc_xgb)
plt.plot(gmd, err_x)
plt.title("Зависимость значения погрешности модели от глубины деревьев")
plt.xlabel("Max_depth, глубина дерева")
plt.ylabel("Функция потерь RMSLE")
plt.show()
# Проверка точности с помощью RMSLE
scc_xgb = mean_squared_log_error(yTest, tt)
print("Погрешность модели", scc_xgb)
print("Модель 1, точность", scc, "Модель 2, точность", scc_xgb)