def add_fold_features_arr(X_tr, y_tr, X_val):
    y_tr = y_tr.copy()
    y_tr.columns = ["arr_delay"]

    carrier_mean = y_tr.join(
        X_tr[['airline']], how='right'
    ).groupby('airline')['arr_delay'].mean()

    X_tr['carrier_arr_delay_mean'] = X_tr['airline'].map(carrier_mean)
    X_val['carrier_arr_delay_mean'] = (
        X_val['airline'].map(carrier_mean).fillna(carrier_mean.mean())
    )

    origin_mean = y_tr.join(
        X_tr[['origin_airport']], how='right'
    ).groupby('origin_airport')['arr_delay'].mean()

    X_tr['origin_arr_delay_mean'] = X_tr['origin_airport'].map(origin_mean)
    X_val['origin_arr_delay_mean'] = (
        X_val['origin_airport'].map(origin_mean).fillna(origin_mean.mean())
    )

    arr_hour_mean = y_tr.join(
        X_tr[['arr_hour']], how='right'
    ).groupby('arr_hour')['arr_delay'].mean()

    X_tr['arr_hour_delay_mean'] = X_tr['arr_hour'].map(arr_hour_mean)
    X_val['arr_hour_delay_mean'] = (
        X_val['arr_hour'].map(arr_hour_mean).fillna(arr_hour_mean.mean())
    )

    return X_tr, X_val
