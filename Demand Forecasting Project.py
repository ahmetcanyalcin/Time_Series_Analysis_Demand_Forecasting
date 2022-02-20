###############################
# Importing libraries
###############################
import pandas as pd
import numpy as np
import datetime as dt
import os

from lightgbm import LGBMRegressor
from mlxtend.evaluate import accuracy_score

from helpers.eda import *
from helpers.data_prep import *
from final_hw.fhw_helpers.f_h_helpers import *
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
from matplotlib import pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
warnings.filterwarnings('ignore')

"""
# güzel bir eda örneği
# from dataprep.datasets import load_dataset
# from dataprep.eda import create_report
# from dataprep.eda import plot, plot_correlation, plot_missing
# create_report(df).show_browser()

###############################
# Reading Data from SQL
###############################
# # SQL query for data
# df = connect_to_SQL()
# joblib.dump(df, "final_hw/raw_data.pkl") #sql table saved as pickle
#
# joblib.dump(df, "final_hw/raw_data_backup.pkl") # backup


# 01.01.2017 - 09.08.2021 sabahı itibariyle satış datasıdır.
"""  # some notes about df
#################################
# Loading the Pickle
#################################
df = joblib.load("final_hw/raw_data.pkl")


df = df.drop("stok_adi", axis=1)
df = df.drop("cari_isim", axis=1)
# new column names
df.columns = ['stock_code', 'store', 'date', 'customer_code', \
              'quantity', 'unit_price', 'amount']
df = df.drop("customer_code", axis=1)
df = df.drop("unit_price", axis=1)
df = df.drop("amount", axis=1)



#################################
# EDA
#################################
df.head()
df.info()
#  #   Column         Non-Null Count    Dtype
# ---  ------         --------------    -----
#  0   stock_code     1523761 non-null  object
#  1   store          1523761 non-null  object
#  2   date           1523761 non-null  datetime64[ns]
#  3   customer_code  1523761 non-null  object
#  4   quantity       1523761 non-null  float64
#  5   unit_price     1523761 non-null  float64
#  6   amount         1523761 non-null  float64
# dtypes: datetime64[ns](1), float64(3), object(3)
# memory usage: 81.4+ MB

df = df.groupby(['store', 'stock_code', 'date']).agg({"quantity": "sum"})
df.reset_index(inplace=True)
df.shape

df.head()

# duplice 73 veri görünüyordu inceledim ve farklı
# günlerde aynı müşteriye ait aynı üründen alım olduğunu gördüm
# x = pd.DataFrame(df[df.duplicated() == True])
#
# x.to_excel("ddd.xlsx")

check_df(df)
# checking the descriptive statistics

df.describe([0.05, 0.1, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

#          quantity
# count 1238426.000
# mean        8.028
# std        81.318
# min         0.050
# 5%          1.000
# 10%         1.000
# 25%         1.000
# 50%         2.000
# 75%         6.000
# 90%        15.000
# 95%        25.000
# 99%        80.000
# max     21600.000

# quan_99 = df[df["quantity"] > df["quantity"].quantile(0.99)]
# quan_99.to_excel("qun.xlsx")


min_date, max_date = df['date'].min(), df['date'].max()
# Out[]: (Timestamp('2017-01-02 00:00:00'), Timestamp('2021-08-09 00:00:00'))


# TODO: quantile ve treshold problemine geri dönülecek


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def grab_outliers(dataframe, col_name, index=False, q1=0.05, q3=0.95):
    """
    we can reach the outlier values directly

    Parameters
    ----------
    dataframe
    col_name
    index

    Returns
    -------

    """
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        return dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    else:
        return dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


# outlier_index = grab_outliers(df, "quantity", q1=0.01, q3=0.99, index=True)
#
# len(outlier_index)
#
# outlier_index.to_excel("test.xlsx")

df.groupby("stock_code").agg({"quantity": "mean"})

df.groupby("stock_code").agg({"stock_code": "count"})

# Kaç store var? (satış noktası)
df[["store"]].nunique()
# store    4
# dtype: int64


# Kaç item var?
df[["stock_code"]].nunique()

# stock_code    19420

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["stock_code"].nunique()
# store
# 1    11172
# 2    12015
# 3     8687
# 7     7027
# Name: stock_code, dtype: int64

# Peki her store'da satış adetleri

df.groupby(["store"]).agg({"quantity": ["sum"]})
#          quantity
#               sum
# store
# 1     5834901.000
# 2      562099.860
# 3     2788173.500
# 7      756496.000

# mağaza-item kırılımında satış istatistikleri
df_ozet = df.groupby(["store", "stock_code"]).agg({"quantity": ["nunique", "sum", "mean", "median", "std"]})
df_ozet2 = df.groupby(["stock_code"]).agg({"quantity": ["nunique", "sum", "mean", "median", "std"]})

df_ozet2.to_excel("final_hw/df_ozet.xlsx")

#####################################################
# FEATURE ENGINEERING
#####################################################

########################
# Date Features
########################
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df['quarter'] = df.date.dt.quarter
    df["is_wknd"] = (df['date'].dt.weekday >= 5).astype(int)
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df


df.head()
df = create_date_features(df)

df.to_excel("satislar.xlsx")
df.groupby(["year", "stock_code"]).agg({"quantity": "sum"})
df.pivot_table(values="quantity", index="stock_code", columns="year", fill_value=0, aggfunc="sum")

# year                     2017    2018  2019   2020  2021
# stock_code
# 2001.MH 02564           2.000   0.000     0  0.000 0.000
# 2001.MH 04077           0.000   0.000     0  1.000 0.000
# 2001.MH 2507            1.000   0.000     0  0.000 0.000
# 2001.MH 2510            1.000   0.000     0  0.000 0.000
# 2001.MH 2516            2.000   0.000     0  0.000 0.000
#                        ...     ...   ...    ...   ...
# YS.VOLKSWAGEN.PU 936 X  2.000   0.000     0  0.000 0.000
# YS.VOLVO.20879812       1.000   0.000     3  0.000 0.000
# YS.WAHLE.0001          34.000   8.000     0  0.000 0.000
# YS.WAHLEN.0001         59.000 128.000    98 28.000 0.000
# YS.WOLKWAGEN.71115562A  0.000   0.000     1  0.000 0.000
# [19420 rows x 5 columns]

# todo: ürünlerin satış frekansına göre eleme yapabiliriz bunu kontrol edelim


# Şu an ay bilgisi olduğu mesela store-item-month kırılımında satış istatistiklerini görebiliriz.
df.groupby(["store", "stock_code", "month"]).agg({"quantity": ["sum", "mean", "median", "std"]})



########################
# Random Noise
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# Lag/Shifted Features
########################

df.sort_values(by=['store', 'stock_code', 'date'], axis=0, inplace=True)

check_df(df)
df.head()

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['quantity_lag_' + str(lag)] = dataframe.groupby(["store", "stock_code"])['quantity'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

check_df(df)

df["quantity"].isnull().sum()


########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "stock_code"])['quantity']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546, 720])
df.tail()


########################
# Exponentially Weighted Mean Features
########################

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "stock_code"])['quantity'].transform(
                    lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.6, 0.4]
lags = [59, 61, 91, 98, 105, 112, 119, 126, 182, 364, 546, 728]

df = ewm_features(df, alphas, lags)

########################
# One-Hot Encoding
########################
df.head()
df.columns
df = pd.get_dummies(df, columns=['store', 'day_of_week', 'month', 'year'])

joblib.dump(df, "final_hw/df_int.pkl")

###############################
# Importing libraries
###############################
import pandas as pd
import numpy as np
import datetime as dt
import os
from helpers.eda import *
from helpers.data_prep import *
from final_hw.fhw_helpers.f_h_helpers import *
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
from matplotlib import pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from lightgbm import LGBMModel

df = joblib.load("final_hw/df.pkl")
df = joblib.load("final_hw/df_int.pkl")

df.shape

df['quantity'] = np.log1p(df["quantity"].values)


########################
# Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


########################
# Time-Based Validation Sets
########################

# 2021'in 5. ayına kadar train seti
train = df.loc[(df["date"] < "2021-05-01"), :]

# 2021'nin 5. aydan sonraki 3 aylık dönem validasyon seti.
val = df.loc[(df["date"] >= "2021-05-01") & (df["date"] < "2021-08-01"), :]

cols = [col for col in train.columns if col not in ['date', 'stock_code', "quantity", "year"]]

Y_train = train['quantity']
X_train = train[cols]

Y_val = val['quantity']
X_val = val[cols]

# kontrol
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'l2'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1,
              "force_col_wise": True}

# modelx= LGBMRegressor(
#         n_estimators=100,
#         learning_rate=0.1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         max_depth=10,
#         num_leaves=31,
#         min_child_weight=300)
#
# modelx.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_val, Y_val)],
#            eval_metric='rmse', verbose=1000, early_stopping_rounds=200)
# model.best_params_

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=1000)

joblib.dump(model, "final_hw/model1.pkl")
model = joblib.load("final_hw/model1.pkl")

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)



smape(np.expm1(y_pred_val), np.expm1(Y_val))


from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred_val = np.expm1(y_pred_val)
mean_absolute_error(Y_val, y_pred_val)


X_val_index = X_val.index
pred_df = val.loc[:, ['date', 'quantity']]
pred_df['quantity_pred'] = np.expm1(y_pred_val)


trainx= train.set_index("date")
valx = val.set_index("date")

pred_dfx= pred_df.set_index("date")
pred_dfx = pred_dfx.drop("quantity", axis=1)
pred_dfx["quantity_pred"] = pred_dfx["quantity_pred"].astype("int")
type(valx)
pred_dfx.columns = ["quantity"]


def plot_time_series(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train.plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

trainx = train[["date", "quantity"]]
valx = val[["date", "quantity"]]
plot_time_series(trainx["quantity"], valx["quantity"], pred_dfx["quantity"],"ggg")

Y_train.shape
y_pred_val.shape
df.info


########################
# Değişken önem düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
        return feat_imp
    else:
        print(feat_imp.head(num))
        return feat_imp


plot_lgb_importances(model, num=60)
aaa= plot_lgb_importances(model, num=120)
aaa.to_excel("features.xlsx")
plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

########################
# Final Model
########################

train = df.loc[~df.quantity.isna()]
Y_train = train['quantity']
X_train = train[cols]

test = df.loc[df.quantity.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = model.predict(X_test, num_iteration=model.best_iteration)
