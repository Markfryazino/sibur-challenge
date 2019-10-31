import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def prepare_shifts(train_target, target_column='pet', shift=6):
    data = train_target.copy()
    data['month'] = data.index.month
    for i in range(1, shift + 1):
        data['shifted_' + str(i)] = data[target_column].shift(i)
    y = data[target_column]
    return data


def concat_train_test(shifts, test_target):
    test = test_target.copy()
    test['month'] = test.index.month
    full_data = pd.concat((shifts, test), axis=0, sort=False)
    return full_data


def prepare_daily(data, daily, shift=20, columns=[]):
    daily2 = daily.copy()
    daily2 = daily2.resample('D').ffill()
    for i in range(shift):
        for col in columns:
            data[col + '__' + str(i + 1)] = daily2.loc[data.index - pd.offsets.MonthOffset(1) +
                                                       pd.Timedelta(days=8) - pd.Timedelta(days=i)][col].values
    return data


def prepare_weekly(data, weekly, weeks=4, columns=[]):
    weekly2 = weekly.copy()
    for col in weekly2.columns:
        weekly2[col].fillna(weekly2[col].median(), inplace=True)
    weekly2 = weekly2.resample('D').ffill()
    for week in range(0, weeks):
        for col in columns:
            data[col + '__' + str(week + 1)] = weekly2.loc[data.index - pd.offsets.MonthOffset(1)
                                                           + pd.Timedelta(days=8) - pd.Timedelta(days=7 * week)][
                col].values
            data[col + '__' + str(week + 1)].fillna(data[col + '__' + str(week + 1)].median(), inplace=True)
    return data


def remove_nans(data, shift):
    data = data[data.index[0] + pd.Timedelta(shift, unit='M'):]
    return data


def fit_predict(data, train_end, test_start, model, shift, target='pet'):
    model.fit(data.drop(target, axis=1)[:train_end].values, data[:train_end][target])

    for date in data[test_start:].index:
        dt = date
        for i in range(1, shift + 1):
            if dt.month == 1:
                dt = pd.datetime(dt.year - 1, 12, 1)
            else:
                dt = pd.datetime(dt.year, dt.month - 1, 1)
            data.loc[date, 'shifted_' + str(i)] = data.loc[dt, target]
        data.loc[date, target] = model.predict([data.drop(target, axis=1).loc[date].values])
    return data


def plot(data, train_target, test_start, target='pet'):
    plt.figure(figsize=(15, 7))
    train_target[target].plot(ax=plt.gca(), label='train')
    data.loc[test_start:][target].plot(ax=plt.gca(), label='test')
    plt.legend(loc=0)


def ultimate_pet(train_target, test_target, daily, weekly, model, shift_pet=6, shift_daily=20,
                 shift_weeks=4, daily_columns=['brent_close', 'USDCNY_close']):
    data = prepare_shifts(train_target, 'pet', shift_pet)
    data = concat_train_test(data, test_target)
    data = prepare_daily(data, daily, shift_daily, daily_columns)
    data = prepare_weekly(data, weekly, shift_weeks, weekly.columns)
    data = remove_nans(data, shift_pet)
    data = fit_predict(data, '2015', '2016', model, shift_pet)
    plot(data, train_target, '2016')
    return data


def ultimate_rubber(train_target, test_target, daily, model, shift_rubber=6, shift_daily=20,
                    daily_columns=['brent_close', 'USDCNY_close', 'USDTHB_mid',
                                   'USDIDR_mid', 'USDVND_open']):
    data = prepare_shifts(train_target, 'rubber', shift_rubber)
    data = concat_train_test(data, test_target)
    data = prepare_daily(data, daily, shift_daily, daily_columns)
    data = remove_nans(data, shift_rubber)
    data = fit_predict(data, '2015', '2016', model, shift_rubber, target='rubber')
    plot(data, train_target, '2016', target='rubber')
    return data
