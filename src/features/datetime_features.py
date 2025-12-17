import pandas as pd

def add_datetime_features(X):
    X = X.copy()

    X['scheduled_dep_dt'] = pd.to_datetime(X['flight_date']) + \
        pd.to_timedelta(
            X['scheduled_dep_time']//100*60 + X['scheduled_dep_time']%100,
            unit='m'
        )

    X['scheduled_arr_dt'] = X['scheduled_dep_dt'] + \
        pd.to_timedelta(X['scheduled_elapsed_time'], unit='m')

    X['dep_hour'] = X['scheduled_dep_dt'].dt.hour
    X['dep_weekday'] = X['scheduled_dep_dt'].dt.weekday
    X['month'] = X['scheduled_dep_dt'].dt.month

    X['arr_hour'] = X['scheduled_arr_dt'].dt.hour
    X['arr_weekday'] = X['scheduled_arr_dt'].dt.weekday
    X['arr_day'] = X['scheduled_arr_dt'].dt.day

    X = X.drop(columns=[
        'scheduled_dep_dt',
        'scheduled_arr_dt',
        'scheduled_dep_time',
        'flight_date'
    ])

    return X
