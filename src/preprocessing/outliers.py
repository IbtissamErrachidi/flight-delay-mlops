def filter_outliers(X_train, y_train, X_test, y_test):
    lower = y_train.quantile(0.01)
    upper = y_train.quantile(0.99)

    mask_train = (y_train >= lower) & (y_train <= upper)
    mask_test = (y_test >= lower) & (y_test <= upper)

    X_train = X_train[mask_train].copy()
    y_train = y_train[mask_train].copy()
    X_test = X_test[mask_test].copy()
    y_test = y_test[mask_test].copy()

    return X_train, y_train, X_test, y_test
