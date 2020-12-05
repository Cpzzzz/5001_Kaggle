import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv('./data/train.csv')

holidays = ['20170101', '20170128', '20170129', '20170130', '20170131', '20170404', '20170501', '20170530', '20170701', '20171001', '20171005', '20171028', '20171222', '20171225', '20180101',
            '20180216', '20180217', '20180218', '20180219', '20180405', '20180501', '20180618', '20180701', '20180925', '20181001', '20181017', '20181222', '20181225', ]


def isHoliday(format_date):
    date_str = str(format_date.year) + str(format_date.month).zfill(2) + str(format_date.day).zfill(2)
    if date_str in holidays:
        return 1
    return 0


def isCommute(format_date):
    hour = format_date.hour
    weekday = format_date.isoweekday()
    holiday = isHoliday(format_date)
    if weekday == 6 or weekday == 7 or holiday == 1:
        return 0
    if hour < 8 or hour > 19:
        return 0
    return 1


train['format_date'] = pd.to_datetime(train['date'], format='%d/%m/%Y %H:%M')
train['year'] = train['format_date'].apply(lambda x: x.year)
train['month'] = train['format_date'].apply(lambda x: x.month)
train['day'] = train['format_date'].apply(lambda x: x.day)
train['hour'] = train['format_date'].apply(lambda x: x.hour)
train['weekday'] = train['format_date'].apply(lambda x: x.isoweekday())
train['dayOfYear'] = train['format_date'].apply(lambda x: x.dayofyear)
train['weekOfYear'] = train['format_date'].apply(lambda x: x.weekofyear)
train['ifWeekday'] = train['weekday'].apply(lambda x: 1 if x in range(1, 6) else 0)
train['isHoliday'] = train['format_date'].apply(lambda x: isHoliday(x))
train['isCommute'] = train['format_date'].apply(lambda x: isCommute(x))

X_data = pd.DataFrame({'year': train['year'],
                       'month': train['month'],
                       'day': train['day'],
                       'hour': train['hour'],
                       'weekday': train['weekday'],
                       'dayOfYear': train['dayOfYear'],
                       'weekOfYear': train['weekOfYear'],
                       'ifWeekday': train['ifWeekday'],
                       'isHoliday': train['isHoliday'],
                       'isCommute': train['isCommute'],
                       })

Y_data = pd.DataFrame(train['speed'])

X_train, X_valid, Y_train, Y_valid = train_test_split(X_data, Y_data, test_size=0.2)

xgb = XGBRegressor(n_estimators=800, learning_rate=0.05, min_child_weight=5, max_depth=8)
# xgb.fit(X_train, Y_train)
xgb.fit(X_data, Y_data)
print("Validation:", xgb.score(X_valid, Y_valid))

################################################################################################

test = pd.read_csv('./data/test.csv')

test['format_date'] = pd.to_datetime(test['date'], format='%d/%m/%Y %H:%M')
test['year'] = test['format_date'].apply(lambda x: x.year)
test['month'] = test['format_date'].apply(lambda x: x.month)
test['day'] = test['format_date'].apply(lambda x: x.day)
test['hour'] = test['format_date'].apply(lambda x: x.hour)
test['weekday'] = test['format_date'].apply(lambda x: x.isoweekday())
test['dayOfYear'] = test['format_date'].apply(lambda x: x.dayofyear)
test['weekOfYear'] = test['format_date'].apply(lambda x: x.weekofyear)
test['ifWeekday'] = test['weekday'].apply(lambda x: 1 if x in range(1, 6) else 0)
test['isHoliday'] = test['format_date'].apply(lambda x: isHoliday(x))
test['isCommute'] = test['format_date'].apply(lambda x: isCommute(x))

X_data = pd.DataFrame({'year': test['year'],
                       'month': test['month'],
                       'day': test['day'],
                       'hour': test['hour'],
                       'weekday': test['weekday'],
                       'dayOfYear': test['dayOfYear'],
                       'weekOfYear': test['weekOfYear'],
                       'ifWeekday': test['ifWeekday'],
                       'isHoliday': test['isHoliday'],
                       'isCommute': test['isCommute'],
                       })

predict = xgb.predict(X_data)
sub = pd.DataFrame(test['id'])
sub["speed"] = predict
sub.to_csv('result.csv', index=False)
