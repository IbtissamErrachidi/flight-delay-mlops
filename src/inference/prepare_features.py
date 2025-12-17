def add_features_test(X_train, y_train, X_test):
    y_train = y_train.copy()
    y_train.columns = ["arr_delay"]

    carrier_mean = y_train.join(
        X_train[['airline']], how='right'
    ).groupby('airline')['arr_delay'].mean()

    origin_mean = y_train.join(
        X_train[['origin_airport']], how='right'
    ).groupby('origin_airport')['arr_delay'].mean()

    arr_hour_mean = y_train.join(
        X_train[['arr_hour']], how='right'
    ).groupby('arr_hour')['arr_delay'].mean()

    X_test['carrier_arr_delay_mean'] = (
        X_test['airline'].map(carrier_mean).fillna(carrier_mean.mean())
    )
    X_test['origin_arr_delay_mean'] = (
        X_test['origin_airport'].map(origin_mean).fillna(origin_mean.mean())
    )
    X_test['arr_hour_delay_mean'] = (
        X_test['arr_hour'].map(arr_hour_mean).fillna(arr_hour_mean.mean())
    )

    return X_test
